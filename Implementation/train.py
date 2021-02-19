import model, utils

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as F

from PIL import Image


class Dilation():
    def __init__(self, mode, num_classes=21, init_weights=True, ignore_index=-1, gpu_id=0, print_freq=10, epoch_print=10):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.gpu = gpu_id
        self.print_freq = print_freq
        self.epoch_print = epoch_print

        torch.cuda.set_device(self.gpu)

        self.loss_function = nn.CrossEntropyLoss(ignore_index=self.ignore_index).cuda(self.gpu)
        
        self.mode = mode
        self.front = model.Front_end(self.num_classes).cuda(self.gpu)
        if self.mode == 'front-only': self.context = nn.Identity().cuda(self.gpu)
        elif self.mode == 'context': self.context = model.Context(self.num_classes, init_weights).cuda(self.gpu)
        elif self.mode == 'context-large': self.context = model.Context_Large(self.num_classes, init_weights).cuda(self.gpu)

        self.eps = 1e-10
        self.best_mIoU = 0.


    def train(self, train_data, test_data, save=False, epochs=74, lr=0.01, momentum=0.9, weight_decay=0.0005):
        
        params = list(self.front.parameters()) + list(self.context.parameters())
        optimizer = optim.SGD(params, lr, momentum=momentum, weight_decay=weight_decay)

        self.front.train()
        self.context.train()
        for epoch in range(epochs):
            if epoch % self.epoch_print == 0: print('Epoch {} Started...'.format(epoch+1))
            for i, (X, y) in enumerate(train_data):
                n, c, h, w = y.shape
                y = y.view(n, h, w).type(torch.LongTensor)
                
                X, y = X.cuda(self.gpu, non_blocking=True), y.cuda(self.gpu, non_blocking=True)
                output = self.front(X)
                output = F.resize(output, (h, w), Image.BILINEAR)
                output = self.context(output)

                loss = self.loss_function(output, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i+1) % self.print_freq == 0:
                    test_mIoU, test_loss = self.test(test_data)
                    
                    if epoch % self.epoch_print == 0:
                        state = ('Iteration : {} - Train Loss : {:.6f}, Test Loss : {:.6f}, '
                                 'Test mIoU : {:.4f}').format(i+1, loss.item(), test_loss, 100 * test_mIoU)
                        if test_mIoU > self.best_mIoU:
                            print()
                            print('*' * 35, 'Best mIoU Updated', '*' * 35)
                            print(state)
                            self.best_mIoU = test_mIoU
                            if save:
                                torch.save(self.front.state_dict(), './best_front.pt')
                                if self.mode != 'front-only':
                                    torch.save(self.context.state_dict(), './best_context.pt')
                                print('Saved Best Model')
                            print()
                        else:
                            print(state)


    def test(self, test_data):
        # only consider for batch size 1 on test_data
        tps = torch.zeros(self.num_classes).cuda(self.gpu, non_blocking=True)
        fps = torch.zeros(self.num_classes).cuda(self.gpu, non_blocking=True)
        fns = torch.zeros(self.num_classes).cuda(self.gpu, non_blocking=True)
        losses = list()

        self.front.eval()
        self.context.eval()
        with torch.no_grad():
            for i, (X, y) in enumerate(test_data):
                n, c, h, w = y.shape
                y = y.view(n, h, w).type(torch.LongTensor)
                
                X, y = X.cuda(self.gpu, non_blocking=True), y.cuda(self.gpu, non_blocking=True)
                output = self.front(X)
                output = F.resize(output, (h, w), Image.BILINEAR)
                output = self.context(output)

                loss = self.loss_function(output, y)
                losses.append(loss.item())

                tp, fp, fn = utils.mIoU(output, y, self.num_classes, self.gpu)
                tps += tp
                fps += fp
                fns += fn
                
        self.front.train()
        self.context.train()
        
        mIoU = torch.sum(tps / (self.eps + tps + fps + fns)) / self.num_classes
        return (mIoU.item(), sum(losses)/len(losses))