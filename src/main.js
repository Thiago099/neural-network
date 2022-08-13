import './style.css'

import { network } from './network/network'
import { renderNetwork } from './network/render'

const app = document.querySelector('#app')


var myNetwork = new network(1,2,2,1)

app.appendChild(renderNetwork(myNetwork))

myNetwork.Learn(
  [
    [0], [1],
    [1], [0],
  ])


