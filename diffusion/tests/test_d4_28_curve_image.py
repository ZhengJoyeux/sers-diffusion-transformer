import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from src.curve_image_codec import DEFAULT_CURVE, CurveImageCodec, fit_curve_state
from src.model_builder import build_diffusion_model
from src.conditional_prior_residual import ConditionalPriorResidualBank

torch.set_num_threads(2)


def tiny_config(height=64, length=64, timesteps=8, sampling=4):
    state = dict(version='d4_28_curve_image_v1', fit_on='train_only', height=height,
                 width=length, sequence_length=length, measured_length=length,
                 y_min=-4.0, y_max=4.0, axis=(np.arange(length)+600).tolist(),
                 bridge=[dict(vector=[0.0]*14, inverse_rate=[1.0]*length,
                              inverse_scale=0.1, condition_id='test')])
    return dict(model=dict(architecture='dit_curve2d', model_dimension=32,
                  dimension_multipliers=[1,2,4], channels=1, dropout=0.0,
                  dit=dict(hidden_size=32, depth=2, num_heads=4, patch_size=8,
                           gradient_checkpointing=True)),
                curve_image={**DEFAULT_CURVE, 'height':height, 'patch_width':8, 'state':state},
                diffusion=dict(diffusion_steps=timesteps, sampling_steps=sampling,
                    objective='pred_x0', beta_schedule='cosine', auto_normalize=False),
                data=dict(raman_axis_mode='union_with_valid_mask'),
                prior_residual=dict(enabled=True), broad_local_residual=dict(enabled=True),
                conditioning=dict(enabled=True, vector_size=14, prior_spectrum=dict(enabled=True)))


def batch(length=64):
    x = torch.randn(2,1,length)*0.1
    mask = torch.ones_like(x); mask[1,:,length*3//4:]=0
    return x*mask, dict(valid_mask=mask, condition=torch.zeros(2,14),
                       prior_conditioning=torch.sin(torch.arange(length)/7)[None,None].expand(2,-1,-1)*mask)


@pytest.mark.parametrize('height',[64,768])
def test_fixed_coordinate_curve_roundtrip(height):
    codec = CurveImageCodec(tiny_config(height)['curve_image'])
    x = torch.linspace(-2,2,64)[None,None]; mask=torch.ones_like(x)
    restored=codec.decode(codec.render(x,mask),mask)
    error=(restored-x).abs().max()
    assert error < (codec.y_max-codec.y_min)/(height-1)*0.2


def test_same_y_at_different_samples_has_same_pixel_position():
    codec=CurveImageCodec(tiny_config()['curve_image'])
    x=torch.stack([torch.ones(64),torch.ones(64)*2])[:,None]
    image=codec.render(x,torch.ones_like(x))
    assert torch.equal(image[0,:,:,0],image[0,:,:,20])
    assert image[0,0,:,0].argmax()!=image[1,0,:,0].argmax()


def test_out_of_axis_is_not_clipped():
    codec=CurveImageCodec(tiny_config()['curve_image'])
    with pytest.raises(ValueError,match='outside fixed'):
        codec.render(torch.ones(1,1,64)*10,torch.ones(1,1,64))


def test_invalid_tail_does_not_affect_rendering():
    codec=CurveImageCodec(tiny_config()['curve_image']);x,kw=batch()
    a=codec.render(x,kw['valid_mask'])
    b=codec.render(x+(1-kw['valid_mask'])*10000,kw['valid_mask'])
    torch.testing.assert_close(a,b)
    assert not (a*(1-codec.pixel_mask(kw['valid_mask']))).any()


def test_full_height_patch_model_accepts_images_and_unpatchifies():
    model,diff=build_diffusion_model(tiny_config(),64)
    x,kw=batch(); codec=model.codec
    image=codec.render(kw['prior_conditioning'],kw['valid_mask'])
    result=model(image,torch.ones(2,dtype=torch.long),valid_mask=codec.pixel_mask(kw['valid_mask']),
                 condition=kw['condition'],prior_conditioning=image)
    assert result.shape==(2,1,64,64)
    assert not result.any()  # adaLN-Zero
    assert model.patch_embed.in_features==3*64*8
    assert model.final_linear.out_features==64*8
    assert not any(isinstance(m,(torch.nn.Conv1d,torch.nn.Conv2d)) for m in model.modules())
    with pytest.raises((ValueError,TypeError)):
        model(x,torch.ones(2,dtype=torch.long),**kw)


def test_training_really_noises_2d_pixels_and_backpropagates():
    model,diff=build_diffusion_model(tiny_config(),64);x,kw=batch()
    seen=[]
    hook=model.register_forward_pre_hook(lambda m,args:seen.append(tuple(args[0].shape)))
    optimizer=torch.optim.Adam(diff.parameters(),lr=1e-3)
    for _ in range(3):
        optimizer.zero_grad();loss=diff(x,**kw)
        assert torch.isfinite(loss);loss.backward();optimizer.step()
    hook.remove()
    assert seen==[(2,1,64,64)]*3
    for name in ('patch_embed','time_embed','condition_encoder'):
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in getattr(model,name).parameters())
    assert model.blocks[0].qkv.weight.grad.abs().sum()>0
    assert diff.get_latest_loss_components()['curve_image_loss']>0


@pytest.mark.parametrize('sampling',[4,8])
def test_real_2d_ddim_ddpm_sampling_and_ema(sampling):
    from src.ddpm_trainer import ExponentialMovingAverage
    model,diff=build_diffusion_model(tiny_config(sampling=sampling),64);x,kw=batch()
    ema=ExponentialMovingAverage(diff,decay=.995,update_every=1);ema.update(diff)
    states=[]
    hook=ema.ema_model.model.register_forward_pre_hook(lambda m,args:states.append(tuple(args[0].shape)))
    out=ema.ema_model.sample(batch_size=2,**kw);hook.remove()
    assert out.shape==x.shape and torch.isfinite(out).all()
    assert not (out*(1-kw['valid_mask'])).any()
    assert states==[(2,1,64,64)]*sampling
    assert ema.ema_model.last_curve_images.shape==(2,1,64,64)
    assert ema.ema_model._sampling_valid_mask is None


def test_checkpoint_configuration_and_weights_roundtrip(tmp_path):
    config=tiny_config();model,diff=build_diffusion_model(config,64);x,kw=batch()
    with torch.no_grad(): model.final_linear.weight.normal_(0,.01)
    path=tmp_path/'checkpoint.pt';torch.save(dict(configuration=config,diffusion_state=diff.state_dict()),path)
    loaded=torch.load(path,weights_only=False);_,other=build_diffusion_model(loaded['configuration'],64)
    other.load_state_dict(loaded['diffusion_state']);diff.eval();other.eval()
    torch.manual_seed(123);a=diff.sample(batch_size=2,**kw)
    torch.manual_seed(123);b=other.sample(batch_size=2,**kw)
    torch.testing.assert_close(a,b)


@pytest.mark.parametrize('mode',['broad_local','full_residual'])
def test_exact_bridge_matches_original_bank_and_holdouts_do_not_fit_axes(mode):
    rng=np.random.default_rng(2);length=64
    values=rng.normal(0,.03,(40,length)).astype(np.float32)+np.sin(np.arange(length)/9).astype(np.float32)
    masks=np.ones_like(values);masks[20:,48:]=0;values*=masks
    ids=['a']*20+['b']*20;train=list(range(12))+list(range(20,32))
    prior=dict(enabled=True,prior_method='pca_reconstruction',pca_max_components=3,
               training_cross_fit=dict(enabled=True,method='leave_one_out'),
               raman_variance_equalization=dict(enabled=True,equalization_power=.5))
    vectors=np.zeros((40,14),np.float32);vectors[20:,0]=1
    bank=ConditionalPriorResidualBank(prior_configuration=prior,broad_local_configuration=dict(enabled=True,residual_mode=mode))
    bank.fit(values,valid_masks=masks,condition_ids=ids,training_indices=train,raman_shift=np.arange(length)+600)
    targets,bases=bank.transform_training_aware_with_conditioning(values,valid_masks=masks,condition_ids=ids,training_indices=train)
    args=dict(bank=bank,full=values,bases=bases,masks=masks,conditions=vectors,condition_ids=ids,
              training_indices=train,padded_length=length,configuration=tiny_config())
    state=fit_curve_state(**args)
    altered=values.copy();altered[12:20]+=1000;altered[32:]+=1000
    assert state==fit_curve_state(**{**args,'full':altered})
    codec=CurveImageCodec({**DEFAULT_CURVE,'height':64,'state':state})
    t=torch.from_numpy(targets)[:,None];b=torch.from_numpy(bases)[:,None]
    m=torch.from_numpy(masks)[:,None];c=torch.from_numpy(vectors)
    full=codec.full_from_residual(t,b,c,m)
    np.testing.assert_allclose(full[:,0].numpy(),values,atol=3e-6)
    torch.testing.assert_close(codec.residual_from_full(full,b,c,m),t,atol=2e-5,rtol=2e-5)


def test_sampling_images_saved_as_actual_float_tensors(tmp_path):
    from src.curve_image_outputs import save_curve_batch
    _,diff=build_diffusion_model(tiny_config(),64);x,kw=batch()
    diff.sample(batch_size=2,**kw);save_curve_batch(diff,tmp_path,0)
    np.testing.assert_array_equal(np.load(tmp_path/'raw_diffusion_001.npz')['image'],diff.last_curve_images[0].numpy())
    assert (tmp_path/'raw_diffusion_001.png').exists()
    assert 'before downstream' in (tmp_path/'column_diagnostics.jsonl').read_text()


def test_no_1d_noise_masquerading_as_2d_diffusion():
    _,diff=build_diffusion_model(tiny_config(),64);x,kw=batch()
    with pytest.raises(ValueError,match='2D diffusion noise'):
        diff.p_losses(x,torch.ones(2,dtype=torch.long),noise=torch.randn_like(x),**kw)


def test_non_multiple_width_padding_and_partial_patch_mask():
    config=tiny_config(length=64);state=config['curve_image']['state'];state['width']=80
    model,diff=build_diffusion_model(config,64);x,kw=batch()
    loss=diff(x,**kw);assert torch.isfinite(loss)
    out=diff.sample(batch_size=2,**kw)
    assert out.shape==x.shape and not diff.last_curve_images[...,64:].any()


def test_real_input_audit_saves_json_and_reconstructs_without_axis_refit(tmp_path):
    from src.curve_image_audit import audit_curve_inputs
    _,diff=build_diffusion_model(tiny_config(),64);x,kw=batch()
    full=diff.codec.full_from_residual(x,kw['prior_conditioning'],kw['condition'],kw['valid_mask'])
    result=audit_curve_inputs(diff,full=full[:,0].numpy(),bases=kw['prior_conditioning'][:,0].numpy(),
        masks=kw['valid_mask'][:,0].numpy(),training_indices=np.array([0]),validation_indices=np.array([1]),
        directory=tmp_path)
    assert result['splits']['validation']['max_error_in_pixels']<.25
    assert (tmp_path/'curve_roundtrip_audit.json').exists()


def test_2d_invalid_pixels_cannot_change_valid_prediction():
    model,_=build_diffusion_model(tiny_config(),64);model.eval();x,kw=batch()
    with torch.no_grad():
        model.final_linear.weight.normal_(0,.02)
        for block in model.blocks: block.adaLN_modulation[-1].weight.normal_(0,.02)
    pixels=model.codec.pixel_mask(kw['valid_mask']);image=model.codec.render(x,kw['valid_mask'])
    t=torch.ones(2,dtype=torch.long)
    a=model(image,t,valid_mask=pixels,condition=kw['condition'],prior_conditioning=image)
    b=model(image+(1-pixels)*10000,t,valid_mask=pixels,condition=kw['condition'],
            prior_conditioning=image-(1-pixels)*10000)
    torch.testing.assert_close(a,b)
