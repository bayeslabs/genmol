#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[test], labels[test])
    acc_test = accuracy(output[test], labels[test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

