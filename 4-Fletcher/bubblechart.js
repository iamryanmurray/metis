(function() {
	var width = 600,
	height = 600;

	var svg = d3.select("#chart")
	.append("svg")
	.attr("height",height)
	.attr("width",width)
	.append("g")
	.attr("transform","translate(0,0)")

	var radiusScale = d3.scaleSqrt().domain([83,586]).range([20,80])
	// the simulation is a colleciton of forces where
	// we want the circles to go and how they interact
	var simulation = d3.forceSimulation()
		.force("x",d3.forceX(width/2).strength(0.05))
		.force("y",d3.forceY(height/2).strength(0.05))
		.force("collide",d3.forceCollide(function(d) {
			return radiusScale(d.count) + 1;
		}))

	var color = d3.scaleOrdinal(d3.schemeCategory20c);

	d3.queue()
	.defer(d3.csv,"topics_with_counts.csv")
	.await(ready)

	function ready(error,datapoints){
		
		var nodes = svg.append("g")
		.attr("class","nodes")
		.selectAll("circle")
		.data(datapoints)
		.enter()
		.append("circle")
		                .style("fill",function(d) { return color(d.topic)})

		.attr("class","node")
		.attr("r",function(d) {
			return radiusScale(d.count)
		});


		var labels = svg.append("g")
                .attr("class", "labels")
                .selectAll("text")
                .data(datapoints)
                .enter()
                .append("text")
 
                .text(function(d) {
                    return d.topic;
                });






	simulation.nodes(datapoints)
		.on("tick",ticked)

	function ticked(){
		nodes
			.attr("cx", function(d) {
				return d.x
			})
			.attr("cy",function(d) {
				return d.y
			})
	}



}
})();