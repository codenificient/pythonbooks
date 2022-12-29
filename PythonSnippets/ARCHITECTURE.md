# Requirements

* Must accept queries or categories for areas of interest
* Must Accept sources if I want to limit to specific publications
* Should integrate with a news api to retrieve top headlines for each category/query
* Should integrate with Slack (where we will receive the suggested articles)
* Should provide functionality to post separate categories to separate channels
* Each Slack post should include the article, publication, description a link to the article

# Design

* News Help Interface
  * Connect to API
  * Query the top headlines based on config
* Slack API Interface / Facade
  * Connect to the API
  * Structure data in desired message format
  * Send messages to a specified channel
* Query object
  * Encapsulate all query logic and perform data checks if needed
  * Map a query to a channel
  * Map query to the name of our result set
* Config
  * House all constants / settings
  * contain an iterable of query objects
* Runner
  * Use the objects to implement the behavior we want and execute the program

# Happy Path

1. Use config file to get a set of queries we want to execute