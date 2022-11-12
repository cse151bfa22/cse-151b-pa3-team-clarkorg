# CSE151B PA3 Memo
## Test Bleu Values
| Task       | Bleu4             | Bleu1              |
| ---------- | ----------------- | ------------------ |
| task-1     | 4.053804507889456 | 62.42328337537673  |
| task-1.3.5 | 1.967342247221346 | 57.433474539429405 |

## Captions Generated
### Task 1
#### Temp = 0.4
| Caption                                                      | Img_id | Good |
| ------------------------------------------------------------ | ------ | ---- |
| a herd  of zebra zebra standing in grass . trees in them .   | 251476 | yes  |
| a man is on a couch in a a bedroom .                         | 520401 | yes  |
| a man  player holding a ball from the field                  | 305206 | yes  |
| a  baseball playing a ball ball with a game . spectators watching . | 374342 | no   |
| a are a  clothing are a man standing on a phone looking      | 3382   | no   |
| a bowl  plate of broccoli of vegetables and out on the . a of a window . a fork | 571279 | no   |
#### Deterministic = On
| Caption                                                      | Img_id |
| ------------------------------------------------------------ | ------ |
| a herd of zebras  zebra grazing in grass . trees in them .   | 251476 |
| a can be see the  screen boy 's on the couch                 | 520401 |
| a baseball of a  baseball throwing a baseball game .         | 305206 |
| a baseball player  is getting a swing swing to . the crowd watches . | 374342 |
| a of people  sitting a shop with umbrellas man to cell . .   | 3382   |
| a bowl plate of  broccoli of vegetables and out on front . a of a window of a fork | 571279 |
#### Temp = 0.0001
| Caption                                                      | Img_id |
| ------------------------------------------------------------ | ------ |
| a zebra of zebra  zebra grazing in grass . trees in them .   | 251476 |
| a man is in a  floor playing a living room .                 | 520401 |
| a baseball is a  ball on a field field .                     | 305206 |
| a baseball is  playing a a baseball ball in to the it . a racket . | 374342 |
| a man of a animals  sitting a bear . a market . to           | 3382   |
| a and with bananas  slices tomatoes  and and and pears . a . | 571279 |
#### Temp = 5
| Caption                                                      | Img_id |
| ------------------------------------------------------------ | ------ |
| drops series  calender buttery volcano hors bluish deal obstacles wedges wiper bordering  candlelit pom swimsuits coned footing seems brings awaiting | 251476 |
| the off-road dorm  mixing guitar problem crooks channels shelter stack ajar tailgate peaceful  letter scrawny window stork grafatti | 520401 |
| different  medicines netting enjoys frisbie train blue hovering gothic dolphins cells  ornaments trophy pedaling hen sunny sixth secures algae sheds | 305206 |
| swivel ironing  overlooked freaky inquisitively cobble buckled eighty bannanas hard-sided  event matter flashlight tries are exposures cellophane longboarder peephole  fudge | 374342 |
| strudel kiwis  folk miller policemen leafless jockey balloon hotplate 7 van mogul slicer  roaming daisies plymstock depicting imposed mist ribbons | 3382   |
| snowsuit cd  chargers 707 zoomed filter foilage goalkeeper mab watchful toothbrush tumble  anytime fourths supported stack overtop winding flush snacking | 571279 |

