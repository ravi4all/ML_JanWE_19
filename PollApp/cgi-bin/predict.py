import SentimentAnalysis

import cgi
import csv

form = cgi.FieldStorage()

text = form.getvalue("text")
pred = SentimentAnalysis.test(text)
# pred = 'Negative'
file = open('count.txt','a')
file.write(pred+'\n')
file.close()

file = open('count.txt','r')
data = file.readlines()
file.close()

p_count = 0
n_count = 0
for i in range(len(data)):
    if data[i] == 'Negative\n':
        n_count += 1
    else:
        p_count += 1

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