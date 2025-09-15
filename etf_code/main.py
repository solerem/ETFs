"""
Application entry point for the Dash ETF Portfolio Optimizer.

This module launches the :class:`dashboard.Dashboard` app. By default it starts
the server with cached/static data reads enabled (``static=True``).

Usage
-----
Run the module as a script to start the server:

.. code-block:: bash

    python run.py

Or import and call :func:`main` from another module:

.. code-block:: python

    from run import main
    main(debug=True)
"""

from dashboard import Dashboard


def main(debug: bool = False) -> None:
    """
    Launch the Dash application.

    :param debug: If ``True``, enable Dash/Flask debug mode (auto-reload, extra logs).
    :type debug: bool
    :returns: ``None``.
    :rtype: None
    """
    db = Dashboard(static=False)
    db.run(debug=debug)


if __name__ == '__main__':
    main(debug=False)
