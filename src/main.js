import './style.css'

import { network } from './network/network'
import { renderNetwork } from './network/render'

const app = document.querySelector('#app')


var myNetwork = new network(2,2)


var x = [[2,1], [1,2]]
var y = [[1,2], [2,1]]


var input = document.getElementById('input')
var output = document.getElementById('output')
var trainElement = document.getElementById('train')
input.addEventListener('keydown', () => {
  if(event.keyCode == 13) {
    localStorage.setItem('input', input.value)
    update()
  }
})
input.value = localStorage.getItem('input')
update()

trainElement.addEventListener('click', () => {
  train()
})


function update() {
  if(input.value == '') return
  output.value = myNetwork.Predict(eval(`[${input.value}]`)).map(x => Math.round(x))
}

function train(){
  for(var i = 0; i < 100000; i++) {
  myNetwork.Learn(x,y)
  }
  update()
}
train()
// stdout
const stdout = document.getElementById('stdout')
stdout.innerHTML = JSON.stringify(myNetwork.layers
  .slice(1)
  .map
  (layer => [layer.weights,layer.biases]), null, 2).replace(/\n/g, '<br>').replace(/ /g, '&nbsp;')


app.appendChild(renderNetwork(myNetwork))
