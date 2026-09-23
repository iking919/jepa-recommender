import logging

def setup_logging(name: str = 'jepa_recommender', level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        # Create console handler
        ch = logging.StreamHandler()
        ch.setLevel(level)

        # Create formatter and add it to the handler
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)

        # Add the handler to the logger
        logger.addHandler(ch)
    return logger