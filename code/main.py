from dashboard import Dashboard

if __name__ == '__main__':
    db = Dashboard(static=False)
    db.run(debug=False)


