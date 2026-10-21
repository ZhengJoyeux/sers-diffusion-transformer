"""Inspect, generate, or evaluate the retained D4.31 epoch-20 model."""
import argparse
import json

from current_diffusion import evaluate, generate, load_selected_checkpoint, selected_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for command in ('inspect', 'generate', 'evaluate'):
        item = commands.add_parser(command)
        item.add_argument('--checkpoint', default=None)
        if command != 'inspect':
            item.add_argument('--real-data', default=None)
            item.add_argument('--condition', default='all')
            item.add_argument('--output', required=True)
        if command == 'generate':
            item.add_argument('--device', default='cuda')
            item.add_argument('--save-stage-trace', action='store_true')
        if command == 'evaluate':
            item.add_argument('--generated', required=True)
            item.add_argument('--reference-split', choices=['validation', 'test'], default='validation')
    args = vars(parser.parse_args())
    command = args.pop('command')
    if command == 'inspect':
        path, state = load_selected_checkpoint(args['checkpoint'])
        print(json.dumps(dict(selection=selected_model(), actual_path=str(path),
                              actual_step=state['step'], weights_verified=True), ensure_ascii=False, indent=2))
    elif command == 'generate':
        generate(**args)
    else:
        evaluate(**args)


if __name__ == '__main__':
    main()
