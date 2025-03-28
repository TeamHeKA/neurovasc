import argparse

from generation.sphn_generation import gen_sphn_kg
from generation.preprocess_data_sphn import preprocess_kg
from models.node_pred_rgcn_sphn import run_rgcn


parser = argparse.ArgumentParser()
parser.add_argument('--num_patients', type=int, default=10000, help='number of patients')
parser.add_argument('--timeOpt', type=str, default='TS_TR', choices=['NT', 'TS', 'TR', 'TS_TR'], help='time information option')
parser.add_argument('--folds', type=int, default=10, help='number of folds for CV')
parser.add_argument('--dr', type=float, default=0.2, help='dropout rate')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--wd', type=float, default=5e-4, help='weight decay')
parser.add_argument('--embed_dim', type=int, default=32, help='embedding dimension')
parser.add_argument('--hidden_dim', type=int, default=32, help='hidden dimension of the model')
args = parser.parse_args()


if __name__ == "__main__":
    # gen_sphn_kg(args.num_patients, args.timeOpt)
    # preprocess_kg(args.num_patients, args.timeOpt)
    run_rgcn(args.num_patients, args.folds, args.timeOpt, 
        dr=args.dr, 
        lr=args.lr, 
        wd=args.wd, 
        embed_dim=args.embed_dim, 
        hidden_dim=args.hidden_dim,
    )
    print("Model training and evaluation completed.")