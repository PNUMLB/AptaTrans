from aptatrans_pipeline import AptaTransPipeline
import argparse

def run(args: dict) -> None:
    """Run the AptaTransPipeline with the given arguments."""
    print('Device:', args['device'])
    pipeline = AptaTransPipeline(
        dim=args['dim'],
        mult_ff=args['mult_ff'],
        dropout=args['dropout'],  # Unified dropout argument
        n_layers=args['n_layers'],
        n_heads=args['n_heads'],
        save_name=args['save_name'],
        load_best_pt=args['load_best_pt'],
        load_best_model=args['load_best_model'],
        device=args['device'],
        seed=args['seed'],
    )
    
    if args['train_mode'] == 'api':
        pipeline.set_data_for_training(batch_size=args['batch_size'])
        pipeline.train(
            epochs=args['epochs'],
            lr=args['lr']
        )
    elif args['train_mode'] == 'aptamer':
        pipeline.set_data_rna_pt(batch_size=args['batch_size'])
        pipeline.pretrain_encoder_aptamer(
            epochs=args['epochs'],
            lr=args['lr']
        )
    elif args['train_mode'] == 'protein':
        pipeline.set_data_protein_pt(batch_size=args['batch_size'])
        pipeline.pretrain_encoder_protein(
            epochs=args['epochs'],
            lr=args['lr']
        )    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a model based on provided arguments.")
    parser.add_argument('--train_mode', type=str, default='api', choices=['aptamer', 'protein', 'api'],
                        help='Training mode: api, aptamer, or protein.')

    parser.add_argument('--n_layers', type=int, default=6, help='Depth (layers) of backbone model.')
    parser.add_argument('--dim', type=int, default=128, help='Input embedding dimension.')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads.')
    parser.add_argument('--mult_ff', type=int, default=2, help='Multiplication factor for feedforward layer.')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate for attention and feedforward layers.')

    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=16, help='Training batch size.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    
    parser.add_argument('--save_name', type=str, default='default', help='Name of the model to save.')
    parser.add_argument('--load_best_pt', action='store_true', help='Load the best pre-trained model if available.')
    parser.add_argument('--load_best_model', action='store_true', help='Load the best trained model if available.')

    parser.add_argument('--seed', type=int, default=1, help='Random seed for reproducibility.')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda', 'cuda:0', 'cuda:1'], help='Device choice: cpu or cuda.')

    try:
        args = parser.parse_args()
        run(vars(args))
    except Exception as e:
        print(f"Error occurred: {e}")
