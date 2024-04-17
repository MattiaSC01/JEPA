from .trainer import Trainer
from ..model.jepa import Jepa


class JepaTrainer(Trainer):
    def __init__(self, alpha: float = 0.99, **kwargs):
        super().__init__(**kwargs)
        assert isinstance(self.model, Jepa), "JepaTrainer expects a Jepa model."
        self.alpha = alpha
    
    def train_step(self, batch: dict) -> dict:
        loss = super().train_step(batch)
        self.model.update_ema(self.alpha)
        return loss
    
    def log_on_train_step(self, losses):
        super().log_on_train_step(losses)
        if self.step % (self.log_interval * 20) == 0:
            norms = self.model.norms_and_similarities()
            self.logger.log_metric(norms["similarity"]["weight_similarity"], "norms/weight_similarity", self.step)
            self.logger.log_metric(norms["similarity"]["bias_similarity"], "norms/bias_similarity", self.step)
            self.logger.log_metric(norms["predictor_norms"]["weight_norm"] / norms["predictor_norms"]["weight_count"], "norms/predictor_weight_norm", self.step)
            self.logger.log_metric(norms["predictor_norms"]["bias_norm"] / norms["predictor_norms"]["bias_count"], "norms/predictor_bias_norm", self.step)
            self.logger.log_metric(norms["encoder_norms"]["weight_norm"] / norms["encoder_norms"]["weight_count"], "norms/encoder_weight_norm", self.step)
            self.logger.log_metric(norms["encoder_norms"]["bias_norm"] / norms["encoder_norms"]["bias_count"], "norms/encoder_bias_norm", self.step)
            self.logger.log_metric(norms["ema_norms"]["weight_norm"] / norms["ema_norms"]["weight_count"], "norms/ema_weight_norm", self.step)
            self.logger.log_metric(norms["ema_norms"]["bias_norm"] / norms["ema_norms"]["bias_count"], "norms/ema_bias_norm", self.step)
