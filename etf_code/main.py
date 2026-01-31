from dashboard import Dashboard
from portfolio import Portfolio


def main(debug: bool = False) -> None:
    Portfolio(refit_weights=False)

    db = Dashboard()
    db.run(debug=debug)


if __name__ == '__main__':
    main(debug=False)
