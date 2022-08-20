
var nodeIndex = 0
const nodeConnection = {}
function renderLayer(layer)
{
    var container = document.createElement("div");
    container.className = "svg-container";

    var svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.style.height = "100vh";
    svg.style.width = "100px";
    

    var group = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    group.setAttribute("width", "60px");
    group.classList.add("node-container");
    group.setAttribute("x", "20px");
    svg.appendChild(group);
    
    
    
    const fullHeight = document.documentElement.clientHeight ;
    const padding = fullHeight / (layer.length+1);
    const maxHeight = fullHeight - padding;

    const groupTreshold = 35
    group.setAttribute("height", fullHeight-padding*2 + groupTreshold*2 + "px");
    group.setAttribute("y", padding - groupTreshold + "px");
    for(var i = 0; i < layer.length; i++) {
        const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        circle.setAttribute("cx", "50px");
        circle.setAttribute("cy", maxHeight / layer.length * i + padding + "px");
        circle.classList.add("node");
        circle.id = "node-" + nodeIndex++;

        circle.addEventListener("mouseover", e=> {
            for(const connection of nodeConnection[circle.id]) {
                connection.classList.add("connection-active")
            }
        })
        circle.addEventListener("mouseout", e=> {
            for(const connection of nodeConnection[circle.id]) {
                connection.classList.remove("connection-active")
            }
        } )
        svg.appendChild(circle);
        // svg text of the id of the node
        // const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
        // text.setAttribute("x", "50px");
        // text.setAttribute("y", maxHeight / layer.length * i + padding + "px");
        // text.setAttribute("fill", "white");
        // text.setAttribute("text-anchor", "middle");
        // text.setAttribute("alignment-baseline", "middle");
        // text.classList.add("node-text");
        // text.innerHTML = circle.id.split("-")[1];
        // svg.appendChild(text);

    }
    container.appendChild(svg);
    return container
}
export function renderNetwork(network)
{
    var result = document.createElement("div");
    const fullWidth = document.documentElement.clientWidth;
    const fullHeight = document.documentElement.clientHeight;
    var lines = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    lines.setAttribute("width", fullWidth + "px");
    lines.setAttribute("height", fullHeight + "px");
    result.appendChild(lines);

    const padding = fullWidth / (network.layers.length+1);
    const maxWidth = fullWidth - padding;
    var previousLayer = null
    var previousX = null
    for(var i = 0; i < network.layers.length; i++) {
        const layer = network.layers[i];
        const layerContainer = renderLayer(layer);
        layerContainer.style.position = "absolute";
        var left = padding -35  + maxWidth / network.layers.length * i
        layerContainer.style.left = left + "px";
        const currentLayer = layerContainer.querySelectorAll(".node");
        const currentX =left +50+"px";
        var maxWeight = Math.max(...layer.weights.map(x => Math.max(...x)))
        if(previousLayer != null) {
            var j = 0
            for(const previousNode of previousLayer) {
                var k = 0
                for(const currentNode of currentLayer) {
                    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
                    if(nodeConnection[previousNode.id] == null) nodeConnection[previousNode.id] = []
                    if(nodeConnection[currentNode.id] == null) nodeConnection[currentNode.id] = []
                    nodeConnection[currentNode.id].push(line);
                    nodeConnection[previousNode.id].push(line);

                    line.setAttribute("x1", previousX);
                    line.setAttribute("y1", previousNode.getAttribute("cy"));
                    line.setAttribute("x2", currentX);
                    line.setAttribute("y2", currentNode.getAttribute("cy"));
                    var currentWeight = layer.weights[k][j]/maxWeight*255;

                    // make current weight between 100 and 200
                    // if(currentWeight < 0)
                    // {
                    //     currentWeight = (currentWeight/4)+100;
                    //     line.setAttribute("stroke", `rgb(${currentWeight},${0},${currentWeight})`);

                    // }
                    // else
                    // {
                        currentWeight = (currentWeight/4)+100;
                        line.setAttribute("stroke", `rgb(${currentWeight},${currentWeight},${currentWeight})`);
                    // }
                    line.classList.add("connection");
                    lines.appendChild(line);
                    k++
                }
                j++
            }
        }
        previousLayer = currentLayer
        previousX = currentX
        result.appendChild(layerContainer);
    }
    return result
}