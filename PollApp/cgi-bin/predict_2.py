import SentimentAnalysis
import cgi
import csv

form = cgi.FieldStorage()

text = form.getvalue("text")
pred = SentimentAnalysis.test(text)

reviews = []
reviews.append([text,pred])

with open('reviews.csv','a', newline='') as file:
    writer = csv.writer(file)
    for data in reviews:
        writer.writerow(data)

dataset = []
with open('reviews.csv','r') as file:
    reader = csv.reader(file)
    for row in reader:
        dataset.append(row)

p_count = 0
n_count = 0
p_review = []
n_review = []
for i in range(len(dataset)):
    if 'Positive' in dataset[i]:
        p_count += 1
        p_review.append(dataset[i])
    else:
        n_count += 1
        n_review.append(dataset[i])


print("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document</title>
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css" integrity="sha384-ggOyR0iXCbMQv3Xipma34MD+dH/1fQ784/j6cY/iJTQUOhcWr7x9JvoRxT2MZw1T" crossorigin="anonymous">
    <script type="text/javascript" src="https://www.gstatic.com/charts/loader.js"></script>    
    <script>
        google.charts.load('current', { 'packages': ['corechart'] });
        google.charts.setOnLoadCallback(drawChart);
        function drawChart() {
            var data = google.visualization.arrayToDataTable([
                ['Task', 'Hours per Day'],
                ['Negative', %d],
                ['Positive', %d]
            ]);
            var chart = new google.visualization.PieChart(document.getElementById('piechart'));
            chart.draw(data);
        }
    </script>
    <style>
        ul {
            list-style: none;
            height: 400px;
            overflow-y : scroll;
        }
        ul li {
            border-bottom: 1px solid gray;
            padding-bottom: 10px;
            margin-bottom: 10px;
        }
    </style>

</head>
"""%(n_count, p_count))

print("""
<body>
<div class="container">
    <div class='row'>
        <div class='col-md-5'>
            <h2>Reviews</h2>
            <ul>
""")

for i in range(len(dataset)):
    if dataset[i][1] == 'Negative':
        color = 'red'
    else:
        color = 'green'
    print("""
        <li style='color:{}'>{}</li>
    """.format(color,dataset[i][0]))

print("""
    </ul>
        </div>
""")
print("""
        <div class='col-md-7'>
            <h1>Prediction is {}</h1>
            <h2>Count is : Positive : {} and Negative : {}</h2>
            <div id='piechart' style="width: 700px; height: 400px;"></div>
        </div>
    </div>
</div>
</body>
</html>
""".format(pred, p_count, n_count))