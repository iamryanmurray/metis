
var width = 750,
    height = 500;
var fill = d3.scale.category20();
// var mydraw = d3.select("#word-cloud").append("svg")


d3.tsv('/umd/up_diff.tsv', function (data) {
    var word_list = [];

    data.forEach(function(row){
        if (row.count > 0) word_list.push({text: row.word, size: Number(row.count)});
    });

    word_list = word_list.sort(function(a,b){
        return (a.size < b.size)? 1:(a.size == b.size)? 0:-1
    }).slice(0,200);

    console.log(word_list)

    var wordScale = d3.scale.linear()
        .range([10,100])
        .domain([d3.min(word_list,function(d) { return d.size; }),
                 d3.max(word_list,function(d) { return d.size; })
               ]);

    d3.layout.cloud().size([width, height])
        .words(word_list)
        .padding(0)
//      .rotate(function() { return ~~(Math.random() * 2) * 90; })
        .font("Impact")
        .fontSize(function(d) { return wordScale(d.size); })
        .on("end", drawCloud)
        .start();
});







function drawCloud(words) {
    d3.select("#word-cloud").append("svg")
    .attr("width", width)
    .attr("height", height)
    .append("g")
    .attr("transform", "translate("+(width / 2)+","+(height / 2)+")")
    .selectAll("text")
    .data(words)
    .enter().append("text")
    .style("font-size", function(d) { return d.size + "px"; })
    .style("font-family", "Impact")
    .style("fill", function(d, i) { return fill(i); })
    .attr("text-anchor", "middle")
    .attr("transform", function(d) {
        return "translate(" + [d.x, d.y] + ")rotate(" + d.rotate + ")";
    })
    .text(function(d) { return d.text; });
}

