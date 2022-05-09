import pytest

from src.compute_probs import build_causal_masked_inputs_and_targets, compute_prior_probs
from src.generation_utils import load_model_and_tokenizer


@pytest.fixture(scope="session")
def bart_large():
    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer("facebook/bart-large")
    return model, tokenizer

@pytest.fixture(scope="module")
def example_inputs_and_target_for_named_entity():
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


# MASKED INPUT: Charlton Athletic have appointed Jose Riga as their new head coach on a <mask>
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
# 80 | ' two' | 0.4591749906539917 <-
# 12 | '-' | 0.9699639678001404 <-
# 180 | 'year' | 0.9490254521369934 <-

# mask prob should the prob of all the tokens combined
def test_compute_prior_probs_returns_the_join_prob(bart_large, example_inputs_and_target_for_named_entity):
    inputs, targets = example_inputs_and_target_for_named_entity

    ne_probs = compute_prior_probs(inputs, targets, bart_large)

    assert ne_probs == [[0.4591749906539917 * 0.9699639678001404 * 0.9490254521369934]]
