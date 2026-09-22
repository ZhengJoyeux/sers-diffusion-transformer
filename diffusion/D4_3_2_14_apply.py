#!/usr/bin/env python3
from __future__ import annotations
import argparse, difflib, sys
from pathlib import Path
import yaml

ROOT=Path.cwd()
P={
'conditional':ROOT/'src/conditional_prior_residual.py',
'loader':ROOT/'src/configuration_loader.py',
'training':ROOT/'scripts/start_ddpm_training.py',
'trainer':ROOT/'src/ddpm_trainer.py',
'generation':ROOT/'scripts/generate_conditional_spectra.py',
'tests':ROOT/'tests/test_conditional_prior_residual.py',
'base_config':ROOT/'config/ddpm_training_d4_3_2_13_medium.yaml',
}
NEW=ROOT/'config/ddpm_training_d4_3_2_14_cross_fitted_pca_medium.yaml'
PATCH=ROOT/'D4_3_2_14_cross_fitted_pca_formal.patch'

def rep(t,o,n,label):
    c=t.count(o)
    if c!=1: raise RuntimeError(f'{label}: expected 1 match, got {c}')
    return t.replace(o,n,1)

def ins(t,a,s,label):
    c=t.count(a)
    if c!=1: raise RuntimeError(f'{label}: anchor expected 1 match, got {c}')
    return t.replace(a,s+a,1)

def patch_conditional(t):
    t=rep(t,
'''        self.prior_configuration = dict(prior_configuration)\n        self.broad_local_configuration = dict(broad_local_configuration)\n        self.model_axis: np.ndarray | None = None\n        self.entries: dict[str, ConditionalPriorEntry] = {}\n''',
'''        self.prior_configuration = dict(prior_configuration)\n        self.broad_local_configuration = dict(broad_local_configuration)\n\n        cross_fit = self.prior_configuration.get("training_cross_fit", {}) or {}\n        if not isinstance(cross_fit, dict):\n            raise TypeError("prior_residual.training_cross_fit必须是字典。")\n        self.training_cross_fit_enabled = bool(cross_fit.get("enabled", False))\n        self.training_cross_fit_method = str(\n            cross_fit.get("method", "leave_one_out")\n        ).strip().lower()\n        if self.training_cross_fit_enabled and self.training_cross_fit_method != "leave_one_out":\n            raise ValueError("D4.3.2.14目前只支持training_cross_fit.method=leave_one_out。")\n\n        self.model_axis: np.ndarray | None = None\n        self.entries: dict[str, ConditionalPriorEntry] = {}\n        # 仅训练数据准备阶段使用；绝不保存进checkpoint。\n        self._cross_fitted_reference_priors: np.ndarray | None = None\n        self._cross_fitted_training_indices: np.ndarray | None = None\n''','conditional.__init__')
    t=rep(t,
'''        self.model_axis = axis.copy()\n        self.entries = {}\n        all_conditions = sorted(set(str(value) for value in conditions))\n''',
'''        self.model_axis = axis.copy()\n        self.entries = {}\n        cross_fitted_references = (\n            np.zeros_like(values, dtype=np.float32)\n            if self.training_cross_fit_enabled else None\n        )\n        cross_fitted_members = (\n            np.zeros(values.shape[0], dtype=bool)\n            if self.training_cross_fit_enabled else None\n        )\n        all_conditions = sorted(set(str(value) for value in conditions))\n''','conditional.fit init')
    t=rep(t,
'''            outer = _build_outer(self.prior_configuration)\n            outer.fit(training_values, raman_shift=condition_axis)\n            reference = outer.reference_priors_for_spectra(training_values)\n            raw_residuals = (training_values - reference).astype(np.float32)\n            broad_local = BroadLocalResidualDecomposer.from_configuration(\n                self.broad_local_configuration\n            )\n            broad_local.fit(raw_residuals, raman_shift=condition_axis)\n''',
'''            # Final PCA：全部training拟合，保存checkpoint并用于val/test/generation。\n            outer = _build_outer(self.prior_configuration)\n            outer.fit(training_values, raman_shift=condition_axis)\n\n            if self.training_cross_fit_enabled:\n                if group_train.size < 5:\n                    raise ValueError(\n                        f"条件{condition_id}只有{group_train.size}条训练光谱；"\n                        "leave-one-out cross-fit至少要求5条。"\n                    )\n                reference_for_training = np.zeros_like(training_values, dtype=np.float32)\n                for held_position in range(group_train.size):\n                    keep = np.ones(group_train.size, dtype=bool)\n                    keep[held_position] = False\n                    temporary_outer = _build_outer(self.prior_configuration)\n                    temporary_outer.fit(training_values[keep], raman_shift=condition_axis)\n                    reference_for_training[held_position:held_position + 1] = (\n                        temporary_outer.reference_priors_for_spectra(\n                            training_values[held_position:held_position + 1]\n                        )\n                    )\n                assert cross_fitted_references is not None\n                assert cross_fitted_members is not None\n                cross_fitted_references[group_train, :valid_length] = reference_for_training\n                cross_fitted_members[group_train] = True\n            else:\n                reference_for_training = outer.reference_priors_for_spectra(training_values)\n\n            # broad/local也必须在OOF raw residual上拟合。\n            raw_residuals = (training_values - reference_for_training).astype(np.float32)\n            broad_local = BroadLocalResidualDecomposer.from_configuration(\n                self.broad_local_configuration\n            )\n            broad_local.fit(raw_residuals, raman_shift=condition_axis)\n''','conditional.fit core')
    t=rep(t,
'''            self.entries[condition_id] = ConditionalPriorEntry(\n                valid_length=valid_length,\n                number_of_training_spectra=int(group_train.size),\n                outer=outer,\n                broad_local=broad_local,\n            )\n        return self\n\n    def _entry(self, condition_id: str) -> ConditionalPriorEntry:\n''',
'''            self.entries[condition_id] = ConditionalPriorEntry(\n                valid_length=valid_length,\n                number_of_training_spectra=int(group_train.size),\n                outer=outer,\n                broad_local=broad_local,\n            )\n\n        if self.training_cross_fit_enabled:\n            assert cross_fitted_references is not None\n            assert cross_fitted_members is not None\n            if not np.array_equal(np.flatnonzero(cross_fitted_members), np.sort(train)):\n                raise RuntimeError("cross-fit训练索引缓存与training_indices不一致。")\n            self._cross_fitted_reference_priors = cross_fitted_references\n            self._cross_fitted_training_indices = np.sort(train)\n        else:\n            self._cross_fitted_reference_priors = None\n            self._cross_fitted_training_indices = None\n        return self\n\n    def _entry(self, condition_id: str) -> ConditionalPriorEntry:\n''','conditional.fit end')
    methods='''    def training_cross_fit_metadata(self) -> dict[str, Any]:\n        return {\n            "enabled": bool(getattr(self, "training_cross_fit_enabled", False)),\n            "method": str(getattr(self, "training_cross_fit_method", "leave_one_out")),\n        }\n\n    def transform_training_aware_with_conditioning(\n        self, spectra: np.ndarray, *, valid_masks: np.ndarray,\n        condition_ids: Sequence[str], training_indices: Sequence[int],\n    ) -> tuple[np.ndarray, np.ndarray]:\n        \"\"\"Training rows use LOO PCA; val/test rows use final PCA.\"\"\"\n        result, conditioning = self.transform_with_conditioning(\n            spectra, valid_masks=valid_masks, condition_ids=condition_ids\n        )\n        if not bool(getattr(self, "training_cross_fit_enabled", False)):\n            return result, conditioning\n        if self._cross_fitted_reference_priors is None or self._cross_fitted_training_indices is None:\n            raise RuntimeError("cross-fit已启用，但LOO reference缓存不存在。")\n        values = _as_2d_finite(spectra, "spectra")\n        masks = _as_2d_finite(valid_masks, "valid_masks")\n        conditions = self._condition_array(condition_ids, values.shape[0])\n        train = np.asarray(training_indices, dtype=np.int64).reshape(-1)\n        if not np.array_equal(np.sort(train), self._cross_fitted_training_indices):\n            raise ValueError("training_indices与fit()时不一致。")\n        references = self._cross_fitted_reference_priors\n        for condition_id in sorted(set(str(v) for v in conditions[train])):\n            entry = self._entry(condition_id)\n            indices = train[conditions[train] == condition_id]\n            active = values[indices, :entry.valid_length]\n            reference = references[indices, :entry.valid_length]\n            for index in indices:\n                if _valid_prefix_length(masks[index]) != entry.valid_length:\n                    raise ValueError(f"条件{condition_id}的training光谱掩码不一致。")\n            raw = (active - reference).astype(np.float32)\n            broad, _ = entry.broad_local.split_raw_residuals(raw)\n            result[indices, :entry.valid_length] = entry.broad_local.transform_raw_residuals(raw)\n            conditioning[indices, :entry.valid_length] = reference + broad\n        if not np.isfinite(result).all() or not np.isfinite(conditioning).all():\n            raise RuntimeError("cross-fit训练变换产生NaN或无穷值。")\n        return result, conditioning\n\n    def transform_training_aware(\n        self, spectra: np.ndarray, *, valid_masks: np.ndarray,\n        condition_ids: Sequence[str], training_indices: Sequence[int],\n    ) -> np.ndarray:\n        transformed, _ = self.transform_training_aware_with_conditioning(\n            spectra, valid_masks=valid_masks, condition_ids=condition_ids,\n            training_indices=training_indices,\n        )\n        return transformed\n\n'''
    t=ins(t,'    def transform(\n        self,\n        spectra: np.ndarray,\n',methods,'conditional methods')
    t=rep(t,
'''            "version": STATE_VERSION,\n            "enabled": True,\n            "model_axis": self.model_axis.tolist(),\n            "number_of_conditions": len(self.entries),\n''',
'''            "version": STATE_VERSION,\n            "enabled": True,\n            "model_axis": self.model_axis.tolist(),\n            "training_cross_fit": self.training_cross_fit_metadata(),\n            "number_of_conditions": len(self.entries),\n''','conditional state')
    t=rep(t,
'''        instance = cls.__new__(cls)\n        instance.prior_configuration = {}\n        instance.broad_local_configuration = {}\n        instance.model_axis = np.asarray(state.get("model_axis"), dtype=np.float64)\n''',
'''        instance = cls.__new__(cls)\n        instance.prior_configuration = {}\n        instance.broad_local_configuration = {}\n        cross_fit_state = state.get("training_cross_fit", {}) or {}\n        if not isinstance(cross_fit_state, dict):\n            raise ValueError("checkpoint中的training_cross_fit无效。")\n        instance.training_cross_fit_enabled = bool(cross_fit_state.get("enabled", False))\n        instance.training_cross_fit_method = str(\n            cross_fit_state.get("method", "leave_one_out")\n        ).strip().lower()\n        if instance.training_cross_fit_enabled and instance.training_cross_fit_method != "leave_one_out":\n            raise ValueError("checkpoint包含不支持的training cross-fit方法。")\n        instance._cross_fitted_reference_priors = None\n        instance._cross_fitted_training_indices = None\n        instance.model_axis = np.asarray(state.get("model_axis"), dtype=np.float64)\n''','conditional restore')
    t=rep(t,
'''        return {\n            "number_of_conditions": len(self.entries),\n            "valid_lengths": {\n''',
'''        return {\n            "number_of_conditions": len(self.entries),\n            "training_cross_fit": self.training_cross_fit_metadata(),\n            "valid_lengths": {\n''','conditional summary')
    return t

def patch_loader(t):
    block='''    # ------------------------------------------------------------------\n    # D4.3.2.14 training-only PCA cross-fit\n    # ------------------------------------------------------------------\n\n    training_cross_fit = (\n        prior_residual.get(\n            "training_cross_fit",\n            {},\n        )\n        or {}\n    )\n\n    if not isinstance(\n        training_cross_fit,\n        dict,\n    ):\n        raise TypeError(\n            "prior_residual.training_cross_fit必须是字典。"\n        )\n\n    cross_fit_enabled = bool(\n        training_cross_fit.get(\n            "enabled",\n            False,\n        )\n    )\n\n    cross_fit_method = str(\n        training_cross_fit.get(\n            "method",\n            "leave_one_out",\n        )\n    ).strip().lower()\n\n    if (\n        cross_fit_enabled\n        and cross_fit_method != "leave_one_out"\n    ):\n        raise ValueError(\n            "D4.3.2.14目前只支持"\n            "prior_residual.training_cross_fit."\n            "method=leave_one_out。"\n        )\n\n    if (\n        cross_fit_enabled\n        and prior_method != "pca_reconstruction"\n    ):\n        raise ValueError(\n            "prior_residual.training_cross_fit"\n            "要求prior_method=pca_reconstruction。"\n        )\n\n    prior_residual[\n        "training_cross_fit"\n    ] = {\n        "enabled": cross_fit_enabled,\n        "method": cross_fit_method,\n    }\n\n'''
    anchor='''    configuration[\n        "prior_residual"\n    ] = prior_residual\n\n    return prior_residual\n'''
    return ins(t,anchor,block,'configuration_loader')

def patch_training(t):
    old='''        if prior_conditioning_enabled:\n            (\n                spectra_for_model,\n                prior_conditionings_full,\n            ) = conditional_prior_residual_bank.transform_with_conditioning(\n                normalized_full_spectra,\n                valid_masks=valid_masks_on_model_axis,\n                condition_ids=condition_ids_on_spectra,\n            )\n            local_inverse_slopes_full = (\n                conditional_prior_residual_bank\n                .local_inverse_linearization_slopes(\n                    spectra_for_model,\n                    valid_masks=valid_masks_on_model_axis,\n                    condition_ids=condition_ids_on_spectra,\n                )\n            )\n        else:\n            spectra_for_model = conditional_prior_residual_bank.transform(\n                normalized_full_spectra,\n                valid_masks=valid_masks_on_model_axis,\n                condition_ids=condition_ids_on_spectra,\n            )\n'''
    new='''        if prior_conditioning_enabled:\n            (\n                spectra_for_model,\n                prior_conditionings_full,\n            ) = conditional_prior_residual_bank.transform_training_aware_with_conditioning(\n                normalized_full_spectra,\n                valid_masks=valid_masks_on_model_axis,\n                condition_ids=condition_ids_on_spectra,\n                training_indices=training_indices,\n            )\n            local_inverse_slopes_full = (\n                conditional_prior_residual_bank\n                .local_inverse_linearization_slopes(\n                    spectra_for_model,\n                    valid_masks=valid_masks_on_model_axis,\n                    condition_ids=condition_ids_on_spectra,\n                )\n            )\n        else:\n            spectra_for_model = conditional_prior_residual_bank.transform_training_aware(\n                normalized_full_spectra,\n                valid_masks=valid_masks_on_model_axis,\n                condition_ids=condition_ids_on_spectra,\n                training_indices=training_indices,\n            )\n'''
    t=rep(t,old,new,'training transform')
    t=rep(t,
'''        summary = conditional_prior_residual_bank.summary()\n        print("\\n===== D4.2 条件PCA+broad/local先验残差 =====")\n''',
'''        summary = conditional_prior_residual_bank.summary()\n        print("\\n===== D4.2 条件PCA+broad/local先验残差 =====")\n        print(\n            "training PCA cross-fit："\n            f"{summary['training_cross_fit']['enabled']}；"\n            f"method={summary['training_cross_fit']['method']}"\n        )\n''','training summary')
    return t

def patch_trainer(t):
    return rep(t,
'''            should_log = (\n                step % self.log_every_steps == 0\n                or step == self.total_training_steps\n            )\n''',
'''            should_log = (\n                step % self.log_every_steps == 0\n                or step == self.total_training_steps\n                or should_validate\n            )\n''','trainer logging')

def patch_generation(t):
    t=rep(t,
'''    if prior_conditioning_enabled:\n        stage_name = "D4.3.2.10先验条件化残差"\n    elif diversity_enabled:\n''',
'''    cross_fit_metadata = (\n        conditional_prior_bank.training_cross_fit_metadata()\n        if conditional_prior_bank is not None\n        else {"enabled": False, "method": "leave_one_out"}\n    )\n    require_cross_fit_checkpoint = bool(\n        generation.get("require_training_cross_fit_checkpoint", False)\n    )\n    if require_cross_fit_checkpoint and not bool(cross_fit_metadata["enabled"]):\n        raise RuntimeError(\n            "当前生成配置要求D4.3.2.14 cross-fit checkpoint，"\n            "但所加载checkpoint未启用training cross-fit。"\n        )\n    if bool(cross_fit_metadata["enabled"]):\n        stage_name = "D4.3.2.14交叉拟合PCA先验残差"\n    elif prior_conditioning_enabled:\n        stage_name = "D4.3.2.10先验条件化残差"\n    elif diversity_enabled:\n''','generation stage')
    t=rep(t,
'''    print(\n        "本次实际抽取的先验谱输入U-Net："\n        f"{prior_conditioning_enabled}"\n    )\n''',
'''    print(\n        "本次实际抽取的先验谱输入U-Net："\n        f"{prior_conditioning_enabled}"\n    )\n    print(\n        "training PCA cross-fit checkpoint："\n        f"{cross_fit_metadata['enabled']}；"\n        f"method={cross_fit_metadata['method']}"\n    )\n''','generation print')
    t=rep(t,
'''        "prior_spectrum_conditioning_enabled": prior_conditioning_enabled,\n''',
'''        "prior_spectrum_conditioning_enabled": prior_conditioning_enabled,\n        "training_cross_fit": cross_fit_metadata,\n        "require_training_cross_fit_checkpoint": require_cross_fit_checkpoint,\n''','generation manifest')
    return t

def patch_tests(t):
    if 'test_d4_3_2_14_cross_fit_disabled_preserves_legacy_transform' in t:
        raise RuntimeError('D4.3.2.14 tests already present')
    add=r'''\n\ndef test_d4_3_2_14_cross_fit_disabled_preserves_legacy_transform() -> None:\n    spectra, masks, conditions, train, axis = _data()\n    prior, broad = _configuration()\n    prior = dict(prior)\n    prior["training_cross_fit"] = {"enabled": False, "method": "leave_one_out"}\n    bank = ConditionalPriorResidualBank(\n        prior_configuration=prior, broad_local_configuration=broad,\n    ).fit(spectra, valid_masks=masks, condition_ids=conditions,\n          training_indices=train, raman_shift=axis)\n    legacy_r, legacy_b = bank.transform_with_conditioning(\n        spectra, valid_masks=masks, condition_ids=conditions)\n    aware_r, aware_b = bank.transform_training_aware_with_conditioning(\n        spectra, valid_masks=masks, condition_ids=conditions, training_indices=train)\n    np.testing.assert_allclose(aware_r, legacy_r, rtol=0.0, atol=0.0)\n    np.testing.assert_allclose(aware_b, legacy_b, rtol=0.0, atol=0.0)\n\n\ndef test_d4_3_2_14_leave_one_out_training_transform_is_out_of_fold() -> None:\n    spectra, masks, conditions, train, axis = _data()\n    prior, broad = _configuration()\n    prior = dict(prior)\n    prior["training_cross_fit"] = {"enabled": True, "method": "leave_one_out"}\n    bank = ConditionalPriorResidualBank(\n        prior_configuration=prior, broad_local_configuration=broad,\n    ).fit(spectra, valid_masks=masks, condition_ids=conditions,\n          training_indices=train, raman_shift=axis)\n    final_r, final_b = bank.transform_with_conditioning(\n        spectra, valid_masks=masks, condition_ids=conditions)\n    aware_r, aware_b = bank.transform_training_aware_with_conditioning(\n        spectra, valid_masks=masks, condition_ids=conditions, training_indices=train)\n    train = np.asarray(train, dtype=np.int64)\n    non_train = np.asarray(sorted(set(range(spectra.shape[0])) - set(train.tolist())), dtype=np.int64)\n    if non_train.size:\n        np.testing.assert_allclose(aware_r[non_train], final_r[non_train], rtol=0.0, atol=0.0)\n        np.testing.assert_allclose(aware_b[non_train], final_b[non_train], rtol=0.0, atol=0.0)\n    assert float(np.max(np.abs(aware_b[train] - final_b[train]))) > 1.0e-7\n    for index in train:\n        entry = bank._entry(str(conditions[index]))\n        n = entry.valid_length\n        local = entry.broad_local.inverse_local_transform(aware_r[index:index+1, :n])\n        restored = aware_b[index:index+1, :n] + local\n        np.testing.assert_allclose(restored, spectra[index:index+1, :n], rtol=2e-5, atol=2e-6)\n\n\ndef test_d4_3_2_14_checkpoint_final_prior_and_legacy_compatibility() -> None:\n    spectra, masks, conditions, train, axis = _data()\n    prior, broad = _configuration()\n    prior = dict(prior)\n    prior["training_cross_fit"] = {"enabled": True, "method": "leave_one_out"}\n    bank = ConditionalPriorResidualBank(\n        prior_configuration=prior, broad_local_configuration=broad,\n    ).fit(spectra, valid_masks=masks, condition_ids=conditions,\n          training_indices=train, raman_shift=axis)\n    state = bank.state_dict()\n    assert state["training_cross_fit"] == {"enabled": True, "method": "leave_one_out"}\n    assert "_cross_fitted_reference_priors" not in state\n    restored = ConditionalPriorResidualBank.from_state_dict(state)\n    assert restored.training_cross_fit_metadata()["enabled"] is True\n    sampled = restored.sample_generation_conditioning(\n        3, condition_id=str(conditions[int(train[0])]),\n        prior_random_generator=np.random.default_rng(2026),\n        broad_random_generator=np.random.default_rng(2027))\n    assert sampled.shape[0] == 3 and np.isfinite(sampled).all()\n    legacy_state = dict(state)\n    legacy_state.pop("training_cross_fit")\n    legacy = ConditionalPriorResidualBank.from_state_dict(legacy_state)\n    assert legacy.training_cross_fit_metadata() == {"enabled": False, "method": "leave_one_out"}\n'''.replace('\\n','\n')
    return t.rstrip()+add+'\n'

def build_config(base):
    c=yaml.safe_load(base)
    c['project']['name']='d4_3_2_14_cross_fitted_pca_medium'
    c['prior_residual']['training_cross_fit']={'enabled':True,'method':'leave_one_out'}
    c['training']['number_of_epochs']=8
    c['training']['validate_every_epochs']=1
    root='outputs/experiments/d4_3_2_14_cross_fitted_pca_medium'
    c['output'].update({
        'output_directory':root,
        'checkpoint_directory':root+'/checkpoints',
        'log_directory':root+'/logs',
        'generated_spectrum_directory':root+'/generated',
        'preview_plot_directory':root+'/plots'})
    c['generation']['require_training_cross_fit_checkpoint']=True
    head='# D4.3.2.14 Cross-Fitted PCA Training Residual\n# 从头8-epoch medium对照；不要resume D4.3.2.13。\n'
    return head+yaml.safe_dump(c,allow_unicode=True,sort_keys=False)

def ud(path,o,n):
    rel=path.relative_to(ROOT).as_posix()
    return ''.join(difflib.unified_diff(o.splitlines(True),n.splitlines(True),fromfile='a/'+rel,tofile='b/'+rel))

def main():
    ap=argparse.ArgumentParser(); g=ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--check',action='store_true'); g.add_argument('--apply',action='store_true'); a=ap.parse_args()
    for x in P.values():
        if not x.is_file(): raise RuntimeError(f'missing {x}')
    orig={k:v.read_text(encoding='utf-8') for k,v in P.items() if k!='base_config'}
    mod={
      'conditional':patch_conditional(orig['conditional']),
      'loader':patch_loader(orig['loader']),
      'training':patch_training(orig['training']),
      'trainer':patch_trainer(orig['trainer']),
      'generation':patch_generation(orig['generation']),
      'tests':patch_tests(orig['tests']),
    }
    newcfg=build_config(P['base_config'].read_text(encoding='utf-8'))
    if NEW.exists() and NEW.read_text(encoding='utf-8')!=newcfg:
        raise RuntimeError(f'{NEW} exists with different content')
    diffs=''.join(ud(P[k],orig[k],mod[k]) for k in mod)
    diffs+=''.join(difflib.unified_diff([],newcfg.splitlines(True),fromfile='/dev/null',tofile='b/'+NEW.relative_to(ROOT).as_posix()))
    print('D4.3.2.14 check: OK')
    print('files:', ', '.join(str(P[k].relative_to(ROOT)) for k in mod))
    print('new config:', NEW.relative_to(ROOT))
    if a.check:
        print('CHECK PASSED; no files written.')
        return
    for k,t in mod.items(): P[k].write_text(t,encoding='utf-8')
    NEW.write_text(newcfg,encoding='utf-8'); PATCH.write_text(diffs,encoding='utf-8')
    print('APPLY PASSED')
    print('audit patch:', PATCH)

if __name__=='__main__':
    try: main()
    except Exception as e:
        print(f'ERROR: {type(e).__name__}: {e}',file=sys.stderr); raise
