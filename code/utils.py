import torch
import torch.nn as nn
from collections import OrderedDict


### MLP stuffs

class Net(nn.Module):
    def __init__(self, inputdim, outputdim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(inputdim, 800),
            nn.ReLU(),
            nn.Linear(800, 3000),
            nn.ReLU(),
            nn.Linear(3000, 1500),
            nn.ReLU(),
            nn.Linear(1500, outputdim)
        ).to(torch.device("cpu"))

    def forward(self, input_tensor):
            return self.model(input_tensor)


def my_load_model(file, model):
    """
    load pre-trained model into model
    solve pb of keys in dic, module => model
    todo : find why we have to do this ??
    """
    loaded_model = torch.load("model_try1.pt")
    new_model = OrderedDict()
    for k, v in loaded_model.items():
        newk = k.replace("module", "model")
        new_model[newk] = v
    model.load_state_dict(new_model)
    return model


def save_model(file):
    """
    file should be .pt file
    save model into file
    todo : check what happens if file already exists
    """
    torch.save(model.state_dict(), file)



# # draft
# def run(properties, embeddings, k, model):
#
#     # cross-validation
#     logging.info('Training {0} with {1}-fold cross-validation'
#                  .format(self.modelname, self.k))
#     regs = [10**t for t in range(-5, -1)] if self.usepytorch else \
#            [2**t for t in range(-1, 6, 1)]
#     skf = StratifiedKFold(n_splits=self.k, shuffle=True,
#                           random_state=self.seed)
#     scores = []
#
#     for reg in regs:
#         scanscores = []
#         for train_idx, test_idx in skf.split(self.train['X'],
#                                              self.train['y']):
#             # Split data
#             X_train, y_train = self.train['X'][train_idx], self.train['y'][train_idx]
#
#             X_test, y_test = self.train['X'][test_idx], self.train['y'][test_idx]
#
#             # Train classifier
#             if self.usepytorch:
#                 clf = MLP(self.classifier_config, inputdim=self.featdim,
#                           nclasses=self.nclasses, l2reg=reg,
#                           seed=self.seed)
#                 clf.fit(X_train, y_train, validation_data=(X_test, y_test))
#             else:
#                 if self.classifier_config['classifier']['nhid'] == -1:
#                     clf = LogisticRegression(C=reg, random_state=self.seed)
#                     clf.fit(X_train, y_train)
#                 elif self.classifier_config['classifier']['nhid'] == 0:
#                     clf = LogisticRegression(C=reg, random_state=self.seed)
#                     clf.fit(X_train, y_train)
#             score = clf.score(X_test, y_test)
#             scanscores.append(score)
#         # Append mean score
#         scores.append(round(100*np.mean(scanscores), 2))
#
#     # evaluation
#     logging.info([('reg:' + str(regs[idx]), scores[idx])
#                   for idx in range(len(scores))])
#     optreg = regs[np.argmax(scores)]
#     devaccuracy = np.max(scores)
#     logging.info('Cross-validation : best param found is reg = {0} \
#         with score {1}'.format(optreg, devaccuracy))
#
#     logging.info('Evaluating...')
#     if self.usepytorch:
#         clf = MLP(self.classifier_config, inputdim=self.featdim,
#                   nclasses=self.nclasses, l2reg=optreg,
#                   seed=self.seed)
#         clf.fit(self.train['X'], self.train['y'], validation_split=0.05)
#     else:
#         clf = LogisticRegression(C=optreg, random_state=self.seed)
#         clf.fit(self.train['X'], self.train['y'])
#     yhat = clf.predict(self.test['X'])
#
#     # scores = cross_val_score(clf, self.test['X'], self.test['y'], cv=self.k)
#     print("STD DEV : " + str(np.std(scores)))
#     print("MEAN : " + str(np.mean(scores)))
#
#
#     print("____REPORT____")
#     #print("precision : " + str(precision_score(self.test["y"], yhat,  average=None)))
#     #print("recall : " + str(recall_score(self.test["y"], yhat, average=None)))
#     print(classification_report(self.test["y"], yhat))
#
#     testaccuracy = clf.score(self.test['X'], self.test['y'])
#     testaccuracy = round(100*testaccuracy, 2)
#
#     return devaccuracy, testaccuracy, yhat
