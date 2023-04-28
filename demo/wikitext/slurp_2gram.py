import collections as clct
import lightning as pl
import os.path as pth


class Model(pl.LightningModule):
    def __init__(self, fname):
        super().__init__()
        self.fname = fname
        self.line_counter = 0
        self.occur_counter = 0
        self.paragraph_counter = 0
        self.freq = dict()
        self.q = clct.deque(maxlen=2)

        self.lookup = {'\n': 0}
        fname = '%s/vocabulary.txt' % pth.dirname(__file__)
        with open(fname, mode='r') as f:
            for ix, wd in enumerate(f):
                wd = wd.strip()
                self.lookup[wd] = ix

    def txt2ix(self, wd):
        try:
            ix = self.lookup[wd]
        except KeyError:
            ix = self.lookup['<unk>']
        return ix

    def slurp(self, x):
        words = x.split(' ')
        length = len(words)
        self.q.clear()
        for ix in range(length):
            wd = words[ix]
            self.q.append(wd)
            gram2 = tuple(self.q)
            if gram2 not in self.freq:
                self.freq[gram2] = 0
                self.line_counter += 1
            self.freq[gram2] += 1
            self.occur_counter += 1

    def test_step(self, train_batch, batch_idx):
        for pix, paragraph in enumerate(train_batch):
            self.slurp(paragraph)
            self.paragraph_counter += 1
        print(self.paragraph_counter, self.line_counter, self.occur_counter)

    def on_test_end(self) -> None:
        with open('datasets/%s.txt' % self.fname, 'w') as f:
            for k, v in sorted(self.freq.items(), key=lambda item: item[0]):
                f.write('{%s: %0.7f}\n' % (k, v / self.occur_counter))


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
    print('test...')
    trainer.test(Model('frequency-train'), train_loader)
    trainer.test(Model('frequency-valid'), valid_loader)
    trainer.test(Model('frequency-test'), test_loader)
