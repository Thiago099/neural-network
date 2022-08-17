import './style.css'

import { network } from './network/network'
import { renderNetwork } from './network/render'

const app = document.querySelector('#app')


const networkShapeElement = document.getElementById('network-shape')
const xElement = document.getElementById('x')
const yElement = document.getElementById('y') 


var input = document.getElementById('input')
var output = document.getElementById('output')

function model(element, defaultValue, callback)
{
  const result = document.querySelector(element)
  result.value = localStorage.getItem(element) || defaultValue
  result.addEventListener('keydown', (e) => {
    if (e.keyCode === 13) {
      callback(result.value)
      localStorage.setItem(element, result.value)
    }
  })
  return result;
}


model('#input', '1,2', input => {
  update()
})

model('#network-shape', '2,2', networkShape => {
  myNetwork = new network(...eval(`[${networkShape}]`))
})

model('#x', '[1,2], [2,1]', xv => {
  x = eval(`[${xv}]`)
  myNetwork = new network(...eval(`[${networkShapeElement.value}]`))
  train()
})

model('#y', '[2,1], [1,2]', yv => {
  y = eval(`[${yv}]`)
  myNetwork = new network(...eval(`[${networkShapeElement.value}]`))
  train()
})

var myNetwork = new network(...eval(`[${networkShapeElement.value}]`))
var x = eval(`[${xElement.value}]`)
var y = eval(`[${yElement.value}]`)

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
  
  app.innerHTML = "";
  app.appendChild(renderNetwork(myNetwork)) 

}
train()
// stdout
