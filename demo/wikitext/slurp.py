import lightning as pl

from transformers import AutoTokenizer


default_length = 18


class SlurpModel(pl.LightningModule):
    def __init__(self, fname):
        super().__init__()
        self.line_counter = 0
        self.paragraph_counter = 0
        self.context = open('datasets/%s.txt' % fname, 'w')
        # self.dedup = set()
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        def vocab_iter():
            with open('datasets/vocab.txt') as f:
                for x in f:
                    x = x.strip()
                    if len(x) > 0:
                        yield x
        self.lookup = {x: ix for ix, x in enumerate(vocab_iter())}
        self.vocab_size = len(self.lookup)

    def txt2ix(self, wd):
        try:
            ix = self.lookup[str(wd)]
        except KeyError:
            ix = self.lookup['<unk>']
        return ix

    def padleft(self, xs):
        xs = list(xs)
        return [0] * (default_length - len(xs)) + xs

    def slurp(self, x):
        tokens = self.tokenizer.tokenize(x)
        length = len(tokens)
        for ix in range(length):
            if ix < default_length:
                self.context.write('%s\n' % self.padleft([self.txt2ix(elm) for elm in tokens[:ix+1]]))
            else:
                self.context.write('%s\n' % self.padleft([self.txt2ix(elm) for elm in tokens[ix - default_length:ix+1]]))
            self.line_counter += 1
        # for ix in range(length):
        #     token = tokens[ix]
        #     if token not in self.dedup:
        #         self.context.write('%s\n' % token)
        #         self.line_counter += 1
        #         self.dedup.add(token)

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
    print('test...')
    # trainer.test(SlurpModel('vocab-train'), train_loader)
    # trainer.test(SlurpModel('vocab-valid'), valid_loader)
    # trainer.test(SlurpModel('vocab-test'), test_loader)
    trainer.test(SlurpModel('context-train'), train_loader)
    trainer.test(SlurpModel('context-valid'), valid_loader)
    trainer.test(SlurpModel('context-test'), test_loader)
