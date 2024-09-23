from aptatrans_pipeline import AptaTransPipeline
import argparse
import yaml


def run(args):
    with open('./config/'+args.config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print(config)
    pipeline = AptaTransPipeline(
        dim=config['dim'],
        mult_ff=config['mult_ff'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        dropout=config['dropout'],
        load_best_model=config['load_best_model'],
        load_best_pt=config['load_best_pt'],
        device=config['device'],
        seed=args.seed,
    )

    if args.encoder == 'rna':
        pipeline.set_data_rna_pt(batch_size=args.batch_size)
        pipeline.pretrain_encoder_aptamer(epochs=args.epochs, lr=args.lr)
    else:
        pipeline.set_data_protein_pt(batch_size=args.batch_size)
        pipeline.pretrain_encoder_protein(epochs=args.epochs, lr=args.lr)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AptaTrans Pretraining')
    parser.add_argument('--config_file', type=str, default='default.yaml', help='config file path')

    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--seed', type=int, default=1004, help='seed')

    parser.add_argument('--epochs', type=int, default=200, help='epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')

    parser.add_argument('--encoder', type=str, default='rna', choices=['apta', 'prot', 'rna', 'protein'], help='encoder')

    args = parser.parse_args()
    run(args)
