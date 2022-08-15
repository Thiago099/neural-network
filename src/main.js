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
input.addEventListener('keydown', (event) => {
  if(event.keyCode == 13) {
    localStorage.setItem('input', input.value)
    update()
  }
})
input.value = localStorage.getItem('input')

trainElement.addEventListener('click', () => {
  train()
})


function update() {
  if(input.value == '') return
  output.value = myNetwork.Predict(eval(`[${input.value}]`)).map(x => Math.round(x))
}

const stdout = document.getElementById('stdout')
function train(){
  
  myNetwork.Learn(x,y,1000)

  stdout.innerHTML = JSON.stringify(myNetwork.layers
    .slice(1)
    .map(layer => layer.weights), null, 2)
    .replace(/\n/g, '<br>')
    .replace(/ /g, '&nbsp;')
  
  update()
}
train()
// stdout

app.appendChild(renderNetwork(myNetwork))
