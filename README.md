# PREDICTING NBA INJURIES

## Motivation
Player health has become increasingly important in recent years.  A lengthy injury to a star player is hugely detrimental to his team's chances of winning.  As an example, the Warriors' offensive rating this year is at a historically great level of 119.1 when Stephen Curry is playing.   However, without him, the Warriors' rating plummets to 105.3, which would slot them at 4th worst in the league.  In addition to the team's on-court success, a star player missing games is also damaging to the teams' finances.  The casual fan will be less inclined to watch a game on TV or go to a game in-person if they know that Kevin Durant or LeBron James is not playing.  This is the reason why I set out to determine if readily available box-score statistics can be used to predict the likelihood of injuries in upcoming games and discover what factors might contribute to injury.

## Data
I scraped three websites to collect my data: prosportstransactions.com for a comprehensive list of all injuries since the inception of the league; nba.com for gamelogs as well as player attributes like heights and weights; and, lastly, Google Maps to get distances between the various NBA cities.  I acquired twenty years worth of data but decided to limit the dataset to only games since 2013.  That was when the NBA started releasing tracking statistics such as speed and distance ran in a game.  I thought these would be good predictors of injuries since they are indicators of how much stress a player is putting on his body.

## Methods
I used a moving window approach to aggregate the statistics of a player within a specified window and used that aggregated data to predict the likelihood of an injury in the next three games.  After trying aggregation windows of one, two, and three weeks, the latter yielded the best results.  It is important to note that I define an injury as something that was directly attributable to playing basketball.  Therefore, sprained ankles and torn ACL's were counted as injuries whereas flus and illnesses were not.

The data was heavily unbalaned in favor of no-injuries so I tried using SMOTE, random undersampling, and random oversampling to resolve the class imbalance.  Sklearn's auto-balancing (class_weight='balanced') turned out to be the most effective method.

The unbalanced nature of the dataset led the algorithms to correctly predict no-injuries the majority of the time.  This resulted in low false-positive rates and, therefore, slightly misleading high ROC AUC scores.  Consequently, I chose to compare my models using the area under the precision-recall curve.  Of the three models that I tried (random forest, gradient-boosted trees, and logistic regression), random forest was the best performer with a precision-recall AUC score of .508 on the holdout data.

## Insights
Random forest determined that the number of miles a player flew was a useful feature in classifying injuries.  Upon closer investigation, the data showed that players who were injured traveled less in the weeks leading up to the injury compared to players who did not get injured.  This contradicts the claims made by sports scientists who say the dense NBA travel schedule is harming players' health and performances.  It should be noted that I am using Google Maps to determine the distances between cities.  Google Maps give driving distances cities but not the actual flying distances.  

## Future Work
NBA.com recently started publishing hustle stats that track things like charges drawn and loose balls recovered.  I would like to incorporate these stats into my model when more becomes available.  I think these features have good predictive power since these actions almost guarantee impact imparted on the body.
