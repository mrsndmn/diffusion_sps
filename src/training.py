import signal

class GracefulKiller:
  kill_now = False
  def __init__(self):
    signal.signal(signal.SIGINT, self.exit_gracefully)
    signal.signal(signal.SIGTERM, self.exit_gracefully)

  def exit_gracefully(self, signum, frame):
    self.kill_now = True

class DiffusionTraining:
    def prepare_timesteps(self):
        raise NotImplemented

    def training_step(self, *args, **kwargs):
        raise NotImplemented

