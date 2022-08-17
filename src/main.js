import './style.css'

import { network } from './network/network'
import { renderNetwork } from './network/render'

const app = document.querySelector('#app')


const networkShapeElement = document.getElementById('network-shape')
const xElement = document.getElementById('x')
const yElement = document.getElementById('y') 


var input = document.getElementById('input')
var output = document.getElementById('output')

input.addEventListener('keydown', (event) => {
  if(event.keyCode == 13) {
    localStorage.setItem('input', input.value)
    update()
  }
})
input.value = localStorage.getItem('input') || '1,2'

networkShapeElement.addEventListener('keydown', () => {
  if(event.keyCode == 13) {
  localStorage.setItem('networkShape', networkShapeElement.value)
  myNetwork = new network(...eval(`[${networkShapeElement.value}]`))
  }
})
networkShapeElement.value = localStorage.getItem('networkShape') || '2,2'

xElement.addEventListener('keydown', () => {
  if(event.keyCode == 13) {
  localStorage.setItem('x', xElement.value)
  x = eval(`[${xElement.value}]`)
  myNetwork = new network(...eval(`[${networkShapeElement.value}]`))
  train()
  }
})

xElement.value = localStorage.getItem('x') || '[1,2], [2,1]'

yElement.addEventListener('keydown', () => {
  if(event.keyCode == 13) {
  localStorage.setItem('y', yElement.value)
  y = eval(`[${yElement.value}]`)
  myNetwork = new network(...eval(`[${networkShapeElement.value}]`))
  train()
  }
})

yElement.value = localStorage.getItem('y') || '[2,1], [1,2]'



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
