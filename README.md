# Clinical trial matching
A conversational chat service to help query UCI's clinical trial databaase, predominantly built using OpenAI and Langchian with a Streamlit UI. Prototype can be found [here](https://uci-clinical-trial-matching.streamlit.app/).

## Quickstart

1. Install the packages provided in ![requirements.txt](/requirements.txt). You'll also need the latest version of **streamlit**.
2. All the requsite datasets can be found inside _data_.
3. The code that is currently deployed on streamlit is contained in _ui.py_ and _util.py_, with all the versioning controlled by _requirements.txt_ - modify these with care.
4. Set your OPENI_API_KEY as an environment variable:

   ```
   export OPENAI_API_KEY="<your_key_goes_here>"
   ```
   To know more about how to get your OpenAI API key visit this ![link](https://platform.openai.com/docs/api-reference/authentication).
5. Use the below code to run the program (with streamlit UI):

   ```
   streamlit run ui.py
   ```
