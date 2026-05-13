Quail v0.7
(initial spec; outdated)

I want this to be a single agent qualitative analysis harness that makes tool calls against a processed corpus to formulate a response. The tool calls should not be strict JSON output, but rather a code mode + API layer.

Quail v0.7 should be a direct Pi (coding agent) fork with significant modifications. The entire tool layer will change. What is kept will mainly be the core agent loop, broad framework, and UI.

In the pi "/" command options, I want an option to process. Selecting process should start this new thread with a separate agent than the one in the main thread. Its history should not be kept, just a side option of sorts. Process should allow the user to select a file or paste in text and then an agent talks with the user about how they want it processed and then proceeds to process it. Processing should be done by the base Pi coding agent with a separate system prompt. What the user must supply to process is:

1. The dataset, of course
2. Metadata confirmation (these are added as fields/tags automatically). The processing agent should preserve metadata
3. What they want the name of the dataset to be, must be unique
4. Confirmation of amount of entries and processing procedure (Ollama, embedding model, etc.)

Processing should take a text corpus of some amount of entries (such as responses to a survey question) and then use Ollama to embed them, default model is embeddinggemma:latest with 64 batch size. A BM25 preprocessing/index should also be prepared, along with any preprocessing needed for having a simple "contains" search (i.e. does an entry contain this exact string of text). Consider the best way to store datasets and the dataset structure. The processing agent may have to modify a dataset or run commands on it to structure it in the right way. Have a workspace/ folder where it can write any file to achieve this.

There should be a clear progress for each step of processing once the processing command is sent in this processing thread. Once the processing command is sent, there can be no other message sent in that processing thread.

I want you to carefully design the system prompt for this processing agent. Make sure it knows how/where to interact and what to run, and what it *needs* to start a run. Make sure the codebase is designed for seamless interaction from this agent. The processing agent should not only be able to add datasets to Quail via processing, but also remove them.

Let's move on to the system prompt for the main agent. This will include the API/language for its analysis.

# Start system prompt

You are an agent in a qualitative research harness. Your job is to use a specific code-like syntax to search the database to answer user questions. Be thorough. Ask questions to the user if anything is unclear. In your answer to a question, never use internal terms, such as groups or entries. Write your answer in a way that is interpretable to any audience. Back up any claims with evidence: quotes, statistics, or any other form.

The syntax you should use when making a tool/code call is:
$
@"<dataset1>", "<dataset2>"
<code>
$

The $ wrap the call and distinguish it from an actual answer.
Every call should have the @ line. This marks what dataset you are using. Often, only one dataset will be activated.

The user has activated the following dataset(s)
- "<dataset1>", <entries>
- "<dataset2>", <entries>
If a dataset is not activated that the user is referencing, ask the user to activate it.

<code> is the tool call itself. Here is the language you adhere to:

Commands:

retrieve(<location> <amount> in (<filter>) of (<group_expression>))
- <location> is either top, middle, or bottom. top takes the top entries, middle, the median, and bottom, the bottom.
- <amount> is how many to retrieve, the maximum is 20
- returns a list of entry ids

get(<id>)
- returns a dictionary of the evidence id:
    - get(<id>).tags, is a list of tags in the format: ["field1": "tag1", "field2": "tag2"]
        - get(<id>).tags["field1"] returns tag1
    - get(<id>).text, returns the full text of an entry id
    - get(<id>).dataset returns the dataset that this entry corresponds to
- or returns a full group spec if <id> is a group id

get(groups)
- returns a list of all group ids

get((<filter>) distribution of (<group_expression>)
- returns a dictionary of the distribution of the filter out of entries in (<group_expression>):
    - get((<filter>) distribution of (<group_expression>).min, returns the minimum BM25/semantic similarity of an entry in (<group_expression>) based on the filter
    - the other options instead of .min are .q1 (first quartile), .q2 (median), .avg (average), .q3 (third quartile) and .max

get(tags)
- returns a dictionary of tags. Indexing into this with a field gives a list of tags
- example: get(tags)["field1"] would return ["tag1", "tag2"] if tag1 and tag2 are the only possible tags for field1
- note that a dataset may come with preexisting fields/tags (metadata). It could be useful to check this at the start

count(<group_expression>)
- returns the amount of entries/responses in a certain <group_expression>

group(<spec>)
- creates a group based on a spec and returns the group id

tag(<id> with <field> set to <tag>)
- tags an entry <id> with its <field> set to a certain <tag>
- if the <field> has not existed before, a new <field> should be created
- <field> and <tag> are strings: "example field and tag"

untag(<field> from <id>)
- removes a tagged <field> from an entry <id>

Additional information:

(<filter>) is in one of two formats:
1. (BM25: "apple battery car")
2. (embeddings: "that's a banana car")
It specifies what the sorting is, i.e. the top BM25/embedding similarity to what?
Note that for BM25, the string supplied is broken by word into keywords. For example:
(BM25: "apple battery car") would use the keywords ["apple", "battery", "car"]

<spec> is the recipe for a group. It is in the format:
BM25: "<text>" > 5.0, embeddings: "<text>", contains: "<text>", exclude: [<id>, <id>, ...], include: [<id>, <id>, ...], tags: ["field1": "tag1", "field2": "tag2"]
where the BM25 field is a string that is broken into keywords and used in the index. Embeddings is an
Any fields (as in BM25, embedding...) can be omitted as needed
- Note that BM25 scores are raw thresholds, and embeddings are cosine similarity

(<group_expression>) is a combination of group ideas with union/intersection/complementation. For example:
((G1 and G2) or (G3 and not G8)) would only include entires that are in G1 and G2 OR G3 and not in G8.
and/or/not refer to intersection/union/complementation
use parantheses as needed

Additional functionality:

1. variables can be set with: var <name>. Example:
var all_groups = get(groups)
- variables should be snake case
- can add/modify variables as per usual (+=, -=, += concatenates to a string)
- lists can be indexed, such as all_groups[i]

2. "for" can be used for loops, simple example:
for entry in retrieve(...)
    get(entry)

3. "if" can also be used, and else too. Example:
if count(...) >= 5:
    for entry in retrieve(...)
        tag(entry with ...)
- comparison operators are >, <, <=, >=, !=, ==
- strings can be compared with ==

4. in/not in can also be used. And print() too. Example:
if <id> not in retrieve(...):
    print("Outside")
print(get(<id>).text)

5. functions can be embedded in each other
retrieve(<location> <amount> in (<filter>) of (BM25: get(<id>).text))

Final notes:

Note that all groups, tags, and variables within your context are saved across code executions.


# End system prompt


When I said that threads, groups, tags and variables are saved within a thread, they should also be saved in a fork of a run. Understand the design of Pi to make sure this works seamlessly.

A user activates a dataset by typing @"<dataset>" on their end. once @ is typed, processed datasets appear and become filtered out as the user types.

Note that this section of the system prompt:

The user has activated the following dataset(s)
- "<dataset1>", <entries>
- "<dataset2>", <entries>

Should be filled in with what the user has activated. And the amount of entries in both datasets.

The Pi coding agent is in the Quail v0.7 repo. When in the Quail v0.7 directory, I want to be able to type, hatch, and Quail launches.

Instead of the pi load-in screen, I want this:
    ,
 <(')
   (V)\
   ^ ^

 Quail
 escape interrupt · ctrl+c/ctrl+d clear/exit · / commands
 a pi fork


Modify the codebase as needed. I currently have Pi in Quail/ but add any folders/files as needed. Comment your code. Build Quail around the system prompt I have provided. Add a suite of parse errors so the main agent can get feedback if its code used incorrect syntax.
