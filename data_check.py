import pickle
train_path = './data_list/model_loss_dice_opt_adamw/train_list.pkl'
val_path = './data_list/model_loss_dice_opt_adamw/train_list.pkl'

with open(train_path, "rb") as f:
    train_files = pickle.load(f)
    print(train_files)