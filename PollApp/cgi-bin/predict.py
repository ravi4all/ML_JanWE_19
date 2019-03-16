import SentimentAnalysis

import cgi
form = cgi.FieldStorage()

text = form.getvalue("text")
pred = SentimentAnalysis.test(text)

print("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document</title>
</head>
<body>
    <h1>Prediction is {}</h1>
</body>
</html>
""".format(pred))