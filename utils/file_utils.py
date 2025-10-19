import os
import json

def get_output_dir(args):
    # Get final folder from args.dataset_name
    
    if hasattr(args, 'eval_mode') and args.eval_mode is not None:
        base_dir = os.path.basename(os.path.normpath(args.eval_dataset_name))
        output_dir = os.path.join(args.save_dir, base_dir, args.eval_mode, args.wandb_name)
    else:
        base_dir = os.path.basename(os.path.normpath(args.dataset_name))
        output_dir = os.path.join(args.save_dir, base_dir, args.train_mode, args.wandb_name)
    if os.path.exists(output_dir):
        raise ValueError(f'Output dir {output_dir} already exists')
    os.makedirs(output_dir)
    print(f'Created {output_dir}')
    return output_dir

def save_text_samples(all_texts_list, save_path):
    full_text = '\n'.join(all_texts_list)
    with open(save_path, 'w', encoding='utf-8') as fo:
        fo.write(full_text)

def save_json(data, save_path):
    with open(save_path, 'w') as fo:
        json.dump(data, fo)