"""Smoke-test: every src module imports without error."""


def test_import_diagnostics():
    import src.diagnostics


def test_import_checkpoint():
    import src.checkpoint


def test_import_loop():
    import src.loop


def test_import_models_loss():
    import src.models.loss


def test_import_models_jepa():
    import src.models.jepa


def test_import_eval_evaluate():
    import src.eval.evaluate


def test_import_eval_bootstrap():
    import src.eval.bootstrap


def test_import_eval_energy():
    import src.eval.energy
