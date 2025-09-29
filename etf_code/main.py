"""
Application entry point for the Dash Portfolio Optimizer.

This module launches the :class:`dashboard.Dashboard` app. By default it starts
the server with cached/static data reads enabled (``static=True``).

Features
--------
* Supports **ETF** and **Crypto** modes (the mode is chosen inside the UI).
* Reads cached CSVs when constructed with ``static=True`` to avoid network calls.
* Exposes a :func:`main` function so you can embed/run the app from other code.

Usage
-----
Run the module as a script to start the server:

.. code-block:: bash

    python run.py

Or import and call :func:`main` from another module:

.. code-block:: python

    from run import main
    main(debug=True)

Parameters
----------
* ``debug`` (bool): When ``True``, enables Dash/Flask debug mode
  (auto-reload, extra logs).

Notes
-----
* The HTTP host/port are Dash defaults. To customize, pass arguments to
  ``db.run(host=..., port=..., debug=...)`` in :func:`main`.
"""

from dashboard import Dashboard
from portfolio import Portfolio

def main(debug: bool = False) -> None:
    """
    Launch the Dash application.

    Parameters
    ----------
    debug : bool, optional
        If ``True``, enable Dash/Flask debug mode (auto-reload, extra logs).
        Default is ``False``.

    Returns
    -------
    None
    """
    Portfolio(allow_short=False)
    Portfolio(crypto=True)
    db = Dashboard()
    db.run(debug=debug)


if __name__ == '__main__':
    main(debug=False)
