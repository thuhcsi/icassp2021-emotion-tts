from tensorflow import keras


def get_loss(loss):
    if type(loss) == str or type(loss) == dict:
        loss_name, kwargs = loss, {}
        if type(loss) == dict:
            loss_name = loss['loss_name']
            kwargs = loss.pop('loss_name')
        kwargs.setdefault('name', loss_name)

        if loss_name == 'scce':
            kwargs.setdefault('from_logits', True)
            return keras.losses.SparseCategoricalCrossentropy(**kwargs)
        elif loss_name == 'cce':
            kwargs.setdefault('from_logits', True)
            return keras.losses.CategoricalCrossentropy(**kwargs)
        elif loss_name == 'bce':
            kwargs.setdefault('from_logits', True)
            return keras.losses.BinaryCrossentropy(**kwargs)
        elif loss_name == 'mae':
            return keras.losses.MeanAbsoluteError(**kwargs)
        elif loss_name == 'mse':
            return keras.losses.MeanSquaredError(**kwargs)
        elif loss_name == 'focal':
            pass
        else:
            raise ValueError('{} is unsupported loss now'.format(loss))
    if callable(loss):
        return loss
    raise TypeError('type of loss must be str or callable, but {} is found'.format(type(loss)))


def get_metric(metric):
    if type(metric) == str or type(metric) == dict:
        metric_name, kwargs = metric, {}
        if type(metric) == dict:
            metric_name = metric['metric_name']
            kwargs = metric.pop('metric_name')
        metric_name = metric_name.lower()
        kwargs.setdefault('name', metric_name)

        if metric == 'sca':
            return keras.metrics.SparseCategoricalAccuracy(**kwargs)
        elif metric == 'ca':
            return keras.metrics.CategoricalAccuracy(**kwargs)
        elif metric == 'ba':
            return keras.metrics.BinaryAccuracy(**kwargs)
        elif metric == 'recall':
            return keras.metrics.Recall(**kwargs)
        elif metric == 'precision':
            return keras.metrics.Precision(**kwargs)
        elif metric == 'mae':
            return keras.metrics.MeanSquaredError(**kwargs)
        elif metric == 'mse':
            return keras.metrics.MeanAbsoluteError(**kwargs)
        else:
            raise ValueError('{} is unsupported metric now'.format(metric))
    if callable(metric):
        return metric
    raise TypeError('type of metric must be str or callable, but {} is found'.format(type(metric)))


def get_optimizer(optimizer, lr_schedule=0.001, **kwargs):
    if type(optimizer) == str:
        optimizer = optimizer.lower()
        if optimizer == 'adam':
            return keras.optimizers.Adam(learning_rate=lr_schedule, **kwargs)
        elif optimizer == 'sgd':
            return keras.optimizers.SGD(learning_rate=lr_schedule, **kwargs)
        elif optimizer == 'rmsprop':
            return keras.optimizers.RMSprop(learning_rate=lr_schedule, **kwargs)
        else:
            raise ValueError('{} is unsupported optimizer now'.format(optimizer))
    if isinstance(optimizer, keras.optimizers.Optimizer):
        return optimizer
    raise TypeError('type of optimizer must be str or callable, but {} is found'.format(type(optimizer)))


def get_regularizer(reg):
    if reg is None:
        return None
    if type(reg) == str:
        return keras.regularizers.get(reg.lower())
    if type(reg) == dict:
        if 'l1' in reg and 'l2' in reg:
            return keras.regularizers.l1_l2(l1=reg['l1'], l2=reg['l2'])
        elif 'l1' in reg:
            return keras.regularizers.l1(reg['l1'])
        elif 'l2' in reg:
            return keras.regularizers.l2(reg['l2'])
        else:
            raise 'the dict keys for regularizer must be "l1", "l2", or both of them'
    if callable(reg):
        return reg
    raise TypeError(f'type of regularizer must be None, str, dict or callable, but {type(reg)} is found')
