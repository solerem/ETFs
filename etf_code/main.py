from dashboard import Dashboard
from portfolio import Portfolio

if __name__ == '__main__':
    Portfolio(refit_weights=False)

    db = Dashboard()
    db.run(debug=False)
