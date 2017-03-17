
# Home win rate vs. Away win rate in Europe Soccer games
------

## Summary
------
This visualization work compares the winning rate between home and away teams in Europe soccer games during 2008-2016.
Whenever we watch a soccer game, one of the team, if it's away and unfortunately lose the game, it seems more understandable. However, does playing home or away affect result that much? If it does, how the difference between home winning rate and away winning rate among different leagues and how the differences are going?

The project began with [Kaggle soccer dataset](https://www.kaggle.com/hugomathien/soccer) with a sqlite database of 7 tables to involve attributes of players, team and matches. To explore the winning rate of home team and away team, I selected "Match" and "League Table". Because the two tables have results from real matches, sourced from: http://football-data.mx-api.enetscores.com/


## Design

The challenges to make such visualization are challenging:
- Data Manipulation work is needed
- There're 25979 matches from 11 leagues, traditional visualization might lead to overwhelming information

I used backward reasoning to sketch the main frame of visualization work.

Firstly, there are 4 attributes should be included in my visualization, and after some minutes thining I ended up with the structure with: 
- year: time line
- winning rate: numerical percentage
- type of such winning rate: categorical, home or away
- League names: categorical

Then I drew some draft graphes of the chart and determine to use the one which can better deliver the right massage to audiance: Set up time line as x-axis, winning rate as y-axis, League legends at right side and type by color of the points and lines. Inspired by the dimple.js example of [advanced story board](http://dimplejs.org/advanced_examples_viewer.html?id=advanced_storyboard_control) and [Interactive Legends](http://dimplejs.org/advanced_examples_viewer.html?id=advanced_interactive_legends) I decided to use animation for the audiance to select freely the league they want to explore.

Witch such design of graph, I designed the structure of my table and used SQL and python to get, join tables, and clean work, after which I end up with the table I want: No redundant information, only year, league and win-rate for home and away teams.

After coding I get my first version of the visualization work. I modified a little bit from my original design. I notice because if I differentiate type of the winning rate, there gonna be too many colors which makes the chart confusing. After all, color was already used for league, so I can try another encoding. I finally decide do not differentiate type by chart itself because I notice that the home winning rate is always much higher than away winning rate. So why not just add notation on y axis to let audiance know, the upper line is always for percentage of home wins. 

## Feedback
It's very important to get feedback before I publish such visualization. Because there're always blind spots as a designer.

After posted in forum talked with some of my friends I got over 10 advices, 3 of them, I think are most valuable:

* Too many lines at first glance, I totally don't understand what are these lines stand for until I tried to click my mouse and make selections. 
* It's better to click legend to show. Intuitively, it's strange if you show everthing but click is to hide the information. If I were clicking the mouse, I always want to grab the information I needed not trying to hide the information. All web pages nowadays are "click-to-show" so it's a common rule of user experience. After all, "Don't make me think" is very important
* Y axis: to differentiate the type, it's better for you to reverse the order of text. "Away" should be lower than "Home" because it's consistant with the graph. If put "Home" on top, it will confuse audiance and take more time for them to digest what's the graph is talk about.
* Better change your title to a more headline style to EXPLAIN to the reader the message you want to convey.

Therefore, I solved the problems mentioned in feedback by:

* I only show 1 pair of line at beginning, used animation to tell audiance there are actually a lot of line pairs for each league, they can select by legend. One might not know the lines disappeared will be lines of different league at first glimpse, but it does tell there're a lot of lines
* I modified the interaction and make it more intuitively. Audiance can click to show the information they want now. If they want to compare, just select the leagues they want without thininking
* I modified the order of text showed on Y axis
* I changed the title, beside, I add explanation below the chart for reader to know more about my findings

After these work, I have my final version: [index_final.html](http://bl.ocks.org/jiehu567/59d4dd4d8149ef93fb5c8b4503367e14).


## Resources

Data Resource: https://www.kaggle.com/hugomathien/soccer

References:
1) Dimple advanced story board: http://dimplejs.org/advanced_examples_viewer.html?id=advanced_storyboard_control
2) Dimple example - advanced interactive legends:
http://dimplejs.org/advanced_examples_viewer.html?id=advanced_interactive_legends
3) Don't Make Me Think: A Common Sense Approach to Web Usability, 2nd Edition, by Steve Krug

