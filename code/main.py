from dashboard import Dashboard

if __name__ == '__main__':
    db = Dashboard(static=True)
    db.run(debug=False)


