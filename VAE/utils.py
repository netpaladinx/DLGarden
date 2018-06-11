class SessionWithouHooks(object):
    def __init__(self, monitored_session):
        self.monitored_session = monitored_session

    def run(self, fetches, feed_dict=None):
        def step_fn(step_context):
            return step_context.session.run(fetches=fetches, feed_dict=feed_dict)

        return self.monitored_session.run_step_fn(step_fn)