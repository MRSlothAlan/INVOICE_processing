"""
test lightgbm
"""
import lightgbm as lgb
import numpy as np


def main():
    data = np.random.rand(500, 10)
    print(data[0])
    label = np.random.randint(2, size=500)
    train_data = lgb.Dataset(data, label=label)
    validation_data = train_data.create_valid(data=data, label=label)
    param = { 'num_leaves': 31, 'objective': 'binary' }
    param['metric'] = 'auc'

    num_round = 10
    bst = lgb.train(param, train_data, num_round, valid_sets=[validation_data])
    bst.save_model('model.txt')

    bst = lgb.Booster(model_file='model.txt')

    data = np.random.rand(7, 10)
    ypred = bst.predict(data)



if __name__ == "__main__":
    main()