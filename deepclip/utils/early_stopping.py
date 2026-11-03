class EarlyStopping:
    def __init__(self, patience = 5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.improved = False # whether the last call improved best_loss (the trainer saves the checkpoint)
        self.early_stop = False

    def __call__(self, loss):
        self.improved = self.is_improved(loss)

        if self.improved:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def is_improved(self, loss):
        return loss < (self.best_loss - self.min_delta)
