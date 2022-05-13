import pytest

from src.compute_probs import (
    build_causal_masked_inputs_and_targets,
    build_masked_inputs_and_targets,
    compute_prior_probs
)
from src.generation_utils import load_model_and_tokenizer


@pytest.fixture(scope="session")
def bart_large():
    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer("facebook/bart-large")
    return model, tokenizer

@pytest.fixture(scope="module")
def single_causal_entity_data():
    return build_causal_masked_inputs_and_targets({
        "source": "The 58-year-old spent three months in charge of the Addicks at the end of the 2013-14 campaign, keeping the club in the Championship. Since leaving The Valley the Belgian has spent time in charge of Blackpool, Standard Liege and Metz. Riga replaces compatriot Karel Fraeye, who was sacked from his post as interim head coach on Wednesday. Charlton are currently 23rd in the Championship table, three points from safety, and are on a run of 10 games without a win in all competitions. Fraeye was appointed in late October following the departure of Guy Luzon, but only won two of his 14 matches in charge of the first team. In a statement on the club website, Addicks owner Roland Duchatelet admitted the club had made errors in player recruitment and said the board of directors accepted responsibility for \"a disappointing season\". \"It was crucial we dealt with the position of the head coach,\" the Belgian businessman added. \"Jose did an excellent job in his short period with Charlton two seasons ago. He was very popular with supporters and I believe that he will get us back on track.\" Riga won seven of his 16 games during his stint at The Valley in 2014 but left the south-east London club when his contract was not renewed that summer and joined Blackpool. BBC Radio London's Andy Rowley. Charlton fans are increasingly angry with how the club is being run by Roland Duchatelet, who is now onto his sixth head coach since taking over the club in January 2014. There have been a number of recent protests at The Valley aimed at Duchatelet and chief executive Katrien Meire from supporters, who have now come together to form a group called \"Coalition Against Roland Duchatelet\" in an attempt to bring about a sale of the club. Riga has far more managerial experience than his predecessor Karel Fraeye but, given his previous links to Duchatelet and the antipathy towards the board of directors, the appointment could only serve to fan the flames for further supporter unrest.",
        "reference": "Championship strugglers Charlton Athletic have reappointed Jose Riga as head coach on an 18-month deal.",
        "prediction": "Charlton Athletic have appointed Jose Riga as their new head coach on a two-year contract.",
        "entities": [
            # {
            #     "start": 33,
            #     "end": 42,
            #     "label": "Non-hallucinated",
            #     "type": "GPE",
            #     "ent": "Jose Riga"
            # },
            {
                "start": 72,
                "end": 80,
                "label": "Non-factual Hallucination",
                "type": "DATE",
                "ent": "two-year"
            }
        ]
    })

@pytest.fixture(scope="module")
def single_masked_entity_data():
    return build_masked_inputs_and_targets({
        "source": "The 58-year-old spent three months in charge of the Addicks at the end of the 2013-14 campaign, keeping the club in the Championship. Since leaving The Valley the Belgian has spent time in charge of Blackpool, Standard Liege and Metz. Riga replaces compatriot Karel Fraeye, who was sacked from his post as interim head coach on Wednesday. Charlton are currently 23rd in the Championship table, three points from safety, and are on a run of 10 games without a win in all competitions. Fraeye was appointed in late October following the departure of Guy Luzon, but only won two of his 14 matches in charge of the first team. In a statement on the club website, Addicks owner Roland Duchatelet admitted the club had made errors in player recruitment and said the board of directors accepted responsibility for \"a disappointing season\". \"It was crucial we dealt with the position of the head coach,\" the Belgian businessman added. \"Jose did an excellent job in his short period with Charlton two seasons ago. He was very popular with supporters and I believe that he will get us back on track.\" Riga won seven of his 16 games during his stint at The Valley in 2014 but left the south-east London club when his contract was not renewed that summer and joined Blackpool. BBC Radio London's Andy Rowley. Charlton fans are increasingly angry with how the club is being run by Roland Duchatelet, who is now onto his sixth head coach since taking over the club in January 2014. There have been a number of recent protests at The Valley aimed at Duchatelet and chief executive Katrien Meire from supporters, who have now come together to form a group called \"Coalition Against Roland Duchatelet\" in an attempt to bring about a sale of the club. Riga has far more managerial experience than his predecessor Karel Fraeye but, given his previous links to Duchatelet and the antipathy towards the board of directors, the appointment could only serve to fan the flames for further supporter unrest.",
        "reference": "Championship strugglers Charlton Athletic have reappointed Jose Riga as head coach on an 18-month deal.",
        "prediction": "Charlton Athletic have appointed Jose Riga as their new head coach on a two-year contract.",
        "entities": [
            # {
            #     "start": 33,
            #     "end": 42,
            #     "label": "Non-hallucinated",
            #     "type": "GPE",
            #     "ent": "Jose Riga"
            # },
            {
                "start": 72,
                "end": 80,
                "label": "Non-factual Hallucination",
                "type": "DATE",
                "ent": "two-year"
            }
        ]
    })

@pytest.fixture(scope="module")
def long_entity_data():
    return build_causal_masked_inputs_and_targets(
        {
            "source": "The greatest foreign policy disasters have tended to come when the UK has either ignored America - such as when it joined France in invading Suez - or when it has followed the US too blindly, as in the invasion of Iraq, against the warnings of many in Europe. Britain has done better when it has played its traditional role as a bridge between the two continents: seeking to manage America's swings between isolationism and interventionism while at the same time reassuring Europe that the US could be more than just a brash Nato ally. This Atlanticist analysis is, of course, a simplification of a complex relationship, but the point stands: the UK has a unique opportunity by virtue of its history and geography to bring the US and Europe together, not take sides. And yet the election of Donald Trump risks doing just that, driving a wedge between Britain and the EU just when both sides need it least. Since the president-elect's shock election, the British government has staggered, recovered its balance and started making overtures to the emerging administration. Its public statements have been welcoming and warm. Here is Foreign Secretary Boris Johnson's latest description of the president-elect, in a Czech newspaper: \"Donald Trump is dealmaker, he is a guy who believes firmly in values that I believe in too - freedom and democracy. As far as I understand, he is in many aspects a liberal guy from New York.\" This is a long way from the criticisms that he and Prime Minister Theresa May uttered during the campaign. But both are now eating their words without shame, for they believe that it is in the national interest for the government to engage with next president of the United States, whatever his character, temperament and policy agenda. The prime minister's Guildhall speech - arguing that everyone should benefit from globalisation - may have been designed to show British voters that she has an agenda for change. But it also sent a strong signal to Mr Trump that she understood the forces that led to his election and the shared need for the whole world to address them. In contrast, some EU countries have worn their hearts on their sleeve and expressed their dissatisfaction at the election of a man whose opinions they find distasteful. This was the context of the emergency meeting of EU foreign ministers this week, which Boris Johnson dismissed as a \"whinge-o-rama\". Some British diplomats saw the meeting as an attempt by the EU foreign policy chief, Federica Mogherini, to grandstand and use Mr Trump's election to drive forward her own agenda for more EU defence cooperation. She was distinctly sarcastic about Mr Johnson's refusal to attend the meeting, saying it was \"normal\" for a country that had decided to leave the EU not to be interested in its relations with the US. As if that were not enough, Mr Johnson also clashed with his EU counterparts over Turkey, urging them not to lecture Ankara over the death penalty. And some EU leaders will not like Mrs May acknowledging the strength of Mr Trump's anti-globalisation arguments just as they prepare to do electoral battle with populist parties flying the same anti-establishment flag. So on the face of it, Britain is falling out with Europe over Mr Trump just when its overwhelming foreign policy objective should be preparing the way for Brexit, working closely with European allies to find the areas of common ground where a potential deal could be done. This should be a time for cool diplomacy, not hot-headed spats over who is going to which meeting. And all this before the tough decisions come. Responding to an election is one thing. It is quite another to choose sides when there are real policy dilemmas on the table. What happens if Mr Trump makes overtures to Russia and calls for sanctions over Ukraine to be reduced? What happens if Mr Trump ends US support for Syrian rebels and orders the US military to join forces with Russia against the Islamic State group? This was the one issue of substance Messrs Trump and Putin discussed in their telephone call this week. And what happens if Mr Trump is true to his word and reduces the US commitment to Nato? All of these are policy choices that could see the UK not only reverse longstanding positions but also place itself at odds with either the US or the EU. This is the context for Mrs May's trip to Berlin on Friday where she will meet President Obama and four of her most important European counterparts, from Germany, France , Italy and Spain. The challenge for this untested, newly appointed prime minister will be to walk that traditional fine line between the US and Europe. She has to prepare the way for these tough policy decisions, urging Europe to engage with a president-elect who cannot be ignored. But equally she will be under some pressure to make it clear to the travelling Americans that Britain will not roll over and accept automatically whatever foreign policy emerges from a Trump-led Washington. The UK-US relationship is hardly special at the moment. Mr Trump has shown no hesitation at embarrassing Mrs May by giving Nigel Farage a photo-opportunity at Trump Towers, giving the impression that the UKIP leader is filling a vacuum left by the Foreign Office. Mr Obama has chosen to visit Greece and Germany and not the UK on a valedictory European tour and in a news conference described Chancellor Merkel as \"probably... my closest international partner these eight years\". In vain did David Cameron flip those burgers in the Downing Street garden with the president. As the one time US Secretary of State Dean Acheson once wrote: \"Of course a unique relation existed between Britain and America... but unique did not mean affectionate.\" So the task now for the British government is to engage with Mr Trump's incipient administration and do what it can - if anything - to shape and influence his foreign policy before firm positions are established. Officials recognise there is a window of opportunity in the coming weeks that should not be wasted that could set the terms for discussions about trade and security. Mrs May is pushing for an early meeting with Mr Trump before the inauguration in January. The Foreign Office is already holding talks with the Trump team about the possibility of Mr Johnson travelling to Washington to meet the Vice-President-elect, Mike Pence, and potentially others in the Trump team in the next few weeks. Officials insist that these discussions began before Mr Farage began claiming his ambassadorial role with Mr Trump. But at the same time, the government also has to ensure that this process of engagement with the US does not make Brexit harder by losing what little political support it still has in Europe's chancelleries. The former Chancellor George Osborne told ITV this week that Britain's focus should be on Europe. \"For the first time really, the most important decisions over the next few years are going to be about our relationship with Europe, not about our relationship with the United States,\" he said. In the past few months, the votes for Brexit and Donald Trump have overturned many of Britain's traditional foreign policy assumptions. The government needs to repair relations with both the US and the EU while not antagonising both in process. Falling between the two geopolitical stools would leave the UK floundering mid-Atlantic without a paddle.",
            "reference": "In the arc of history, Britain has rarely flourished when it has had to choose between Europe and the United States.",
            "prediction": "The greatest foreign policy disasters have tended to come when the UK has either ignored America - such as when it joined France in invading Suez - or when it has followed the US too blindly, as in the invasion of Iraq.",
            "entities": [
                {
                    "start": 122,
                    "end": 128,
                    "label": "Non-hallucinated",
                    "type": "GPE",
                    "ent": "France"
                }
            ]
        })

@pytest.fixture(scope="module")
def causal_multiple_entity_data():
    return build_causal_masked_inputs_and_targets({
        "source": "The 58-year-old spent three months in charge of the Addicks at the end of the 2013-14 campaign, keeping the club in the Championship. Since leaving The Valley the Belgian has spent time in charge of Blackpool, Standard Liege and Metz. Riga replaces compatriot Karel Fraeye, who was sacked from his post as interim head coach on Wednesday. Charlton are currently 23rd in the Championship table, three points from safety, and are on a run of 10 games without a win in all competitions. Fraeye was appointed in late October following the departure of Guy Luzon, but only won two of his 14 matches in charge of the first team. In a statement on the club website, Addicks owner Roland Duchatelet admitted the club had made errors in player recruitment and said the board of directors accepted responsibility for \"a disappointing season\". \"It was crucial we dealt with the position of the head coach,\" the Belgian businessman added. \"Jose did an excellent job in his short period with Charlton two seasons ago. He was very popular with supporters and I believe that he will get us back on track.\" Riga won seven of his 16 games during his stint at The Valley in 2014 but left the south-east London club when his contract was not renewed that summer and joined Blackpool. BBC Radio London's Andy Rowley. Charlton fans are increasingly angry with how the club is being run by Roland Duchatelet, who is now onto his sixth head coach since taking over the club in January 2014. There have been a number of recent protests at The Valley aimed at Duchatelet and chief executive Katrien Meire from supporters, who have now come together to form a group called \"Coalition Against Roland Duchatelet\" in an attempt to bring about a sale of the club. Riga has far more managerial experience than his predecessor Karel Fraeye but, given his previous links to Duchatelet and the antipathy towards the board of directors, the appointment could only serve to fan the flames for further supporter unrest.",
        "reference": "Championship strugglers Charlton Athletic have reappointed Jose Riga as head coach on an 18-month deal.",
        "prediction": "Charlton Athletic have appointed Jose Riga as their new head coach on a two-year contract.",
        "entities": [
            {
                "start": 33,
                "end": 42,
                "label": "Non-hallucinated",
                "type": "GPE",
                "ent": "Jose Riga"
            },
            {
                "start": 72,
                "end": 80,
                "label": "Non-factual Hallucination",
                "type": "DATE",
                "ent": "two-year"
            }
        ]
    })

@pytest.fixture(scope="module")
def masked_multiple_entity_data():
    return build_masked_inputs_and_targets({
        "source": "The 58-year-old spent three months in charge of the Addicks at the end of the 2013-14 campaign, keeping the club in the Championship. Since leaving The Valley the Belgian has spent time in charge of Blackpool, Standard Liege and Metz. Riga replaces compatriot Karel Fraeye, who was sacked from his post as interim head coach on Wednesday. Charlton are currently 23rd in the Championship table, three points from safety, and are on a run of 10 games without a win in all competitions. Fraeye was appointed in late October following the departure of Guy Luzon, but only won two of his 14 matches in charge of the first team. In a statement on the club website, Addicks owner Roland Duchatelet admitted the club had made errors in player recruitment and said the board of directors accepted responsibility for \"a disappointing season\". \"It was crucial we dealt with the position of the head coach,\" the Belgian businessman added. \"Jose did an excellent job in his short period with Charlton two seasons ago. He was very popular with supporters and I believe that he will get us back on track.\" Riga won seven of his 16 games during his stint at The Valley in 2014 but left the south-east London club when his contract was not renewed that summer and joined Blackpool. BBC Radio London's Andy Rowley. Charlton fans are increasingly angry with how the club is being run by Roland Duchatelet, who is now onto his sixth head coach since taking over the club in January 2014. There have been a number of recent protests at The Valley aimed at Duchatelet and chief executive Katrien Meire from supporters, who have now come together to form a group called \"Coalition Against Roland Duchatelet\" in an attempt to bring about a sale of the club. Riga has far more managerial experience than his predecessor Karel Fraeye but, given his previous links to Duchatelet and the antipathy towards the board of directors, the appointment could only serve to fan the flames for further supporter unrest.",
        "reference": "Championship strugglers Charlton Athletic have reappointed Jose Riga as head coach on an 18-month deal.",
        "prediction": "Charlton Athletic have appointed Jose Riga as their new head coach on a two-year contract.",
        "entities": [
            {
                "start": 33,
                "end": 42,
                "label": "Non-hallucinated",
                "type": "GPE",
                "ent": "Jose Riga"
            },
            {
                "start": 72,
                "end": 80,
                "label": "Non-factual Hallucination",
                "type": "DATE",
                "ent": "two-year"
            }
        ]
    })


# masked_input='Charlton Athletic have appointed Jose Riga as their new head coach on a <mask>'
# target='Charlton Athletic have appointed Jose Riga as their new head coach on a two-year'
# masked_input_tokenized=tensor([    0, 33193,  1054,  8899,    33,  3873,  3071,   248, 11742,    25,
#            49,    92,   471,   704,    15,    10, 50264,     2])
# target_tokenized=tensor([    0, 33193,  1054,  8899,    33,  3873,  3071,   248, 11742,    25,
#            49,    92,   471,   704,    15,    10,    80,    12,   180,     2])
# [   ['<s>', 5.760120984632522e-07],
#     ['Charl', 0.9997900128364563],
#     ['ton', 1.0],
#     [' Athletic', 0.9999984502792358],
#     [' have', 0.9997376799583435],
#     [' appointed', 0.9999973773956299],
#     [' Jose', 0.9999958276748657],
#     [' R', 1.0],
#     ['iga', 0.9999985694885254],
#     [' as', 1.0],
#     [' their', 0.9999995231628418],
#     [' new', 0.9999997615814209],
#     [' head', 0.9999979734420776],
#     [' coach', 0.9999996423721313],
#     [' on', 0.9999997615814209],
#     [' a', 0.9999991655349731],
#     [' two', 0.4591749906539917], <-
#     ['-', 0.9699639678001404], <-
#     ['year', 0.9490254521369934], <-
#     ['</s>', 3.0894573228579247e-06]]
# mask prob should the prob of all the tokens combined
def test_compute_causal_prior_probs(bart_large, single_causal_entity_data):
    inputs, targets, entities = single_causal_entity_data

    ne_probs = compute_prior_probs(inputs, targets, entities, bart_large, verbose=True)
    print(ne_probs)

    assert ne_probs == [(0.4591749906539917 * 0.9699639678001404 * 0.9490254521369934)]

# masked_input='Charlton Athletic have appointed Jose Riga as their new head coach on a <mask> contract.'
# target='Charlton Athletic have appointed Jose Riga as their new head coach on a two-year contract.'
# masked_input_tokenized=tensor([    0, 33193,  1054,  8899,    33,  3873,  3071,   248, 11742,    25,
#            49,    92,   471,   704,    15,    10, 50264,  1355,     4,     2])
# target_tokenized=tensor([    0, 33193,  1054,  8899,    33,  3873,  3071,   248, 11742,    25,
#            49,    92,   471,   704,    15,    10,    80,    12,   180,  1355,
#             4,     2])
# [   ['<s>', 2.129253743987647e-06],
#     ['Charl', 0.9992431402206421],
#     ['ton', 1.0],
#     [' Athletic', 0.9999712705612183],
#     [' have', 0.998350977897644],
#     [' appointed', 0.9999775886535645],
#     [' Jose', 0.9999674558639526],
#     [' R', 0.9999996423721313],
#     ['iga', 0.9999938011169434],
#     [' as', 0.9999996423721313],
#     [' their', 0.999990701675415],
#     [' new', 0.9999980926513672],
#     [' head', 0.9999983310699463],
#     [' coach', 0.9999997615814209],
#     [' on', 0.9999997615814209],
#     [' a', 0.9999985694885254],
#     [' two', 0.4322648048400879],
#     ['-', 0.9040818810462952],
#     ['year', 0.6585601568222046],
#     [' contract', 0.8115606904029846],
#     ['.', 0.9139174818992615],
#     ['</s>', 0.9738051891326904]]
def test_compute_masked_prior_probs(bart_large, single_masked_entity_data):
    inputs, targets, entities = single_masked_entity_data

    ne_probs = compute_prior_probs(inputs, targets, entities, bart_large, verbose=True)
    print(ne_probs)

    assert ne_probs == [0.4322648048400879 * 0.9040818810462952 * 0.6585601568222046]

# masked_input='Charlton Athletic have appointed <mask>'
# target='Charlton Athletic have appointed Jose Riga'
# 0 | '<s>' | 9.380718779539166e-08
# 33193 | 'Charl' | 0.999982476234436
# 1054 | 'ton' | 1.0
# 8899 | ' Athletic' | 0.9999997615814209
# 33 | ' have' | 0.9999910593032837
# 3873 | ' appointed' | 0.9999990463256836
# 3071 | ' Jose' | 0.000604625849518925 <-
# 248 | ' R' | 0.014603929594159126 <-
# 11742 | 'iga' | 0.14980721473693848 <-
# 2 | '</s>' | 0.0783548504114151
# masked_input='Charlton Athletic have appointed Jose Riga as their new head coach on a <mask>'
# target='Charlton Athletic have appointed Jose Riga as their new head coach on a two-year'
# 0 | '<s>' | 5.760120984632522e-07
# 33193 | 'Charl' | 0.9997900128364563
# 1054 | 'ton' | 1.0
# 8899 | ' Athletic' | 0.9999984502792358
# 33 | ' have' | 0.9997376799583435
# 3873 | ' appointed' | 0.9999973773956299
# 3071 | ' Jose' | 0.9999958276748657
# 248 | ' R' | 1.0
# 11742 | 'iga' | 0.9999985694885254
# 25 | ' as' | 1.0
# 49 | ' their' | 0.9999995231628418
# 92 | ' new' | 0.9999997615814209
# 471 | ' head' | 0.9999979734420776
# 704 | ' coach' | 0.9999996423721313
# 15 | ' on' | 0.9999997615814209
# 10 | ' a' | 0.9999991655349731
# 80 | ' two' | 0.4591749906539917
# 12 | '-' | 0.9699639678001404
# 180 | 'year' | 0.9490254521369934
def test_causal_many_entities(bart_large, causal_multiple_entity_data):
    inputs, targets, entities = causal_multiple_entity_data

    ne_probs = compute_prior_probs(inputs, targets, entities, bart_large, verbose=True)
    print(ne_probs)

    assert ne_probs == [
        (0.000604625849518925 * 0.014603929594159126 * 0.14980721473693848),
        (0.4591749906539917 * 0.9699639678001404 * 0.9490254521369934)
    ]

# masked_input='Charlton Athletic have appointed <mask> as their new head coach on a two-year contract.'
# target='Charlton Athletic have appointed Jose Riga as their new head coach on a two-year contract.'
# masked_input_tokenized=tensor([    0, 33193,  1054,  8899,    33,  3873, 50264,    25,    49,    92,
#           471,   704,    15,    10,    80,    12,   180,  1355,     4,     2])
# target_tokenized=tensor([    0, 33193,  1054,  8899,    33,  3873,  3071,   248, 11742,    25,
#            49,    92,   471,   704,    15,    10,    80,    12,   180,  1355,
#             4,     2])
# [   ['<s>', 1.0318922250007745e-06],
#     ['Charl', 0.9992539286613464],
#     ['ton', 1.0],
#     [' Athletic', 0.9997195601463318],
#     [' have', 0.9887463450431824],
#     [' appointed', 0.9999438524246216],
#     [' Jose', 0.000811865902505815], <-
#     [' R', 0.022413266822695732], <-
#     ['iga', 0.2136993259191513], <-
#     [' as', 0.8377609252929688],
#     [' their', 0.9831597208976746],
#     [' new', 0.9968374967575073],
#     [' head', 0.9960565567016602],
#     [' coach', 0.9999699592590332],
#     [' on', 0.9961340427398682],
#     [' a', 0.9999985694885254],
#     [' two', 0.9999712705612183],
#     ['-', 1.0],
#     ['year', 1.0],
#     [' contract', 1.0],
#     ['.', 0.9999665021896362],
#     ['</s>', 0.9990930557250977]]
# masked_input='Charlton Athletic have appointed Jose Riga as their new head coach on a <mask> contract.'
# target='Charlton Athletic have appointed Jose Riga as their new head coach on a two-year contract.'
# masked_input_tokenized=tensor([    0, 33193,  1054,  8899,    33,  3873,  3071,   248, 11742,    25,
#            49,    92,   471,   704,    15,    10, 50264,  1355,     4,     2])
# target_tokenized=tensor([    0, 33193,  1054,  8899,    33,  3873,  3071,   248, 11742,    25,
#            49,    92,   471,   704,    15,    10,    80,    12,   180,  1355,
#             4,     2])
# [   ['<s>', 2.129253743987647e-06],
#     ['Charl', 0.9992431402206421],
#     ['ton', 1.0],
#     [' Athletic', 0.9999712705612183],
#     [' have', 0.998350977897644],
#     [' appointed', 0.9999775886535645],
#     [' Jose', 0.9999674558639526],
#     [' R', 0.9999996423721313],
#     ['iga', 0.9999938011169434],
#     [' as', 0.9999996423721313],
#     [' their', 0.999990701675415],
#     [' new', 0.9999980926513672],
#     [' head', 0.9999983310699463],
#     [' coach', 0.9999997615814209],
#     [' on', 0.9999997615814209],
#     [' a', 0.9999985694885254],
#     [' two', 0.4322648048400879], <-
#     ['-', 0.9040818810462952], <-
#     ['year', 0.6585601568222046], <-
#     [' contract', 0.8115606904029846],
#     ['.', 0.9139174818992615],
#     ['</s>', 0.9738051891326904]]
def test_masked_many_entities(bart_large, masked_multiple_entity_data):
    inputs, targets, entities = masked_multiple_entity_data

    ne_probs = compute_prior_probs(inputs, targets, entities, bart_large, verbose=True)
    print(ne_probs)

    assert ne_probs == [
        (0.000811865902505815 * 0.022413266822695732 * 0.2136993259191513),
        (0.4322648048400879 * 0.9040818810462952 * 0.6585601568222046)
    ]

def test_long_entity(bart_large, long_entity_data):
    inputs, targets, entities = long_entity_data

    ne_probs = compute_prior_probs(inputs, targets, entities, bart_large, verbose=True)
    print(ne_probs)
