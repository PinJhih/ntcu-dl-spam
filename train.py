import dataset

if __name__ == "__main__":
    # load datasets
    data_path = "./data/email_classification.csv"
    train_set, val_set = dataset.load_data(data_path)
    print("Training   samples:", len(train_set))
    print("Validation samples:", len(val_set))
