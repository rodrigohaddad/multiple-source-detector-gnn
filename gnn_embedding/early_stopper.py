class EarlyStopper:
    def __init__(self, patience: int):
        self.patience = patience
        self.val_loss, self.epoch, self.count = 0, 0, 0
        self.stop = False

    def register_val_loss(self, val_loss: float, epoch: int):
        # First epoch
        if self.val_loss == 0:
            self.val_loss = val_loss
            self.epoch = epoch
            self.count += 1
            return

        # Old loss is greater than new loss
        if self.val_loss > val_loss:
            self.count = 0
            self.epoch = epoch
            self.val_loss = val_loss
        # New loss is equal or greater than old loss
        else:
            self.count += 1

    def should_stop(self) -> bool:
        self.stop = self.patience <= self.count
        if self.stop:
            print(f"Early stop due to loss {self.val_loss} reached after {self.epoch} epochs")
        return self.stop
