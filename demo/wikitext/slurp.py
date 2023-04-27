import torch
import lightning as pl


class Model(pl.LightningModule):
    def __init__(self, fname):
        super().__init__()
        self.line_counter = 0
        self.paragraph_counter = 0
        self.context = open('datasets/%s.txt' % fname, 'w')
        self.dedup = set()

    def slurp(self, x):
        words = x.split(' ')
        length = len(words)
        for ix in range(length):
            if ix < 12:
                self.context.write('%s\n' % words[:ix])
            else:
                self.context.write('%s\n' % words[ix - 12:ix])
            self.line_counter += 1
        # for ix in range(length):
        #     wd = words[ix]
        #     if wd not in self.dedup:
        #         self.context.write('%s\n' %wd)
        #         self.line_counter += 1
        #         self.dedup.add(wd)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def test_step(self, train_batch, batch_idx):
        for pix, paragraph in enumerate(train_batch):
            self.slurp(paragraph)
            self.paragraph_counter += 1
        print(self.paragraph_counter, self.line_counter)


if __name__ == '__main__':

    print('loading data...')

    from torch.utils.data import DataLoader
    from torchtext.datasets import WikiText2

    dataset = WikiText2('')

    wiki_train, wiki_valid, wiki_test = dataset
    train_loader = DataLoader(wiki_train, batch_size=1)
    valid_loader = DataLoader(wiki_valid, batch_size=1)
    test_loader = DataLoader(wiki_test, batch_size=1)

    # training
    print('construct trainer...')
    trainer = pl.Trainer(accelerator='cpu', precision=32, max_epochs=1)
    trainer.test(Model('context-train'), train_loader)
    trainer.test(Model('context-valid'), valid_loader)
    trainer.test(Model('context-test'), test_loader)
