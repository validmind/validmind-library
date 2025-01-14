# Copyright © 2023-2024 ValidMind Inc. All rights reserved.
# See the LICENSE file in the root of this repository for details.
# SPDX-License-Identifier: AGPL-3.0 AND ValidMind Commercial

non_test_content_block_html = """
<div
  class="lm-Widget p-Widget jupyter-widget-Collapse-header"
  style="padding: 6px; padding-left: 33px; font-size: 14px"
>
  <span>{content_type} Block: '{content_id}'</i></span>
</div>
"""

test_content_block_html = """
<div>
  <h2>{title}</h2>
  <div style="border: 1px solid #ddd; border-radius: 4px; padding: 10px; margin: 10px 0;">
    {description}
  </div>
</div>

<h4 class="vm_required_context">
  Required Inputs: <span style="font-size: 13px"><i>{required_inputs}</i></span>
</h4>

<div style="display: {table_display};">
  <h4>Parameters:</h4>
  <table class="vm_params_table" style="display: {table_display};">
      <tr>
          <th>Parameter</th>
          <th>Default Value</th>
      </tr>
      {params_table}
  </table>
</div>

<div class="unset">
  <h3>How to Run:</h3>

  <button
      onclick="(() => {{e = document.getElementById('expandable_instructions_{uuid}'); e.style.display === 'none' ? e.style.display = 'block' : e.style.display = 'none'}})()"
  >Show/Hide Instructions</button>

  <div id="expandable_instructions_{uuid}" style="display: {instructions_display};">
  <h4>Code:</h4>
    <pre>
        <code class='language-python'>
import validmind as vm

# inputs dictionary maps your inputs to the expected input names
# keys are the expected input names and values are the actual inputs
# values may be string input_ids or the actual VMDataset or VMModel objects
inputs = {example_inputs}
params = {example_params}

# to run and view the result of this test, run the following code:
result = vm.tests.run_test(
  "{test_id}", inputs=inputs, params=params
)

# To see the result of the test, ensure that you have called `vm.init()` and then run:
result.log()</code>
    </pre>
  </div>
</div>

<style>
h5.vm_required_context {{
    margin-top: 25px;
}}
table.vm_params_table {{
  margin-top: 20px;
  width: 350px;
  border-collapse: collapse;
  border-color: --jp-border-color0;
}}
table.vm_params_table td, table.vm_params_table th {{
  text-align: right;
}}
table.vm_params_table td:first-child, table.vm_params_table th:first-child {{
  text-align: left;
}}
table.vm_params_table th {{
  background-color: --jp-content-color0;
  font-weight: bold;
  font-size: 14px !important;
}}
table.vm_params_table tr:nth-child(even) {{
  background-color: --jp-layout-color1;
}}
table.vm_params_table tr:nth-child(odd) {{
  background-color: --jp-layout-color2;
}}
table.vm_params_table tr:hover {{
  background-color: --jp-layout-color3;
}}
table.vm_params_table td, table.vm_params_table th {{
  padding: 5px;
  border: .8px solid --jp-border-color0;
}}
</style>
"""

python_syntax_highlighting = """
<script defer type="module">
import hljs from 'https://unpkg.com/@highlightjs/cdn-assets@11.9.0/es/highlight.min.js';
import python from 'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/es/languages/python.min.js';

hljs.registerLanguage('python', python);
hljs.highlightAll();
</script>
"""

# have to dynamically load mathjax
math_jax_snippet = """
<script>
window.MathJax = {
    tex2jax: {
        inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
        displayMath: [['$$', '$$'], ['\\[', '\\]']],
        processEscapes: true,
        skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
        ignoreClass: ".*",
        processClass: "math"
    }
};
setTimeout(function () {
    var script = document.createElement('script');
    script.type = 'text/javascript';
    script.src = 'https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-AMS_HTML';
    document.head.appendChild(script);
}, 300);
</script>
"""

failed_content_block_html = """
<div
  class="lm-Widget p-Widget jupyter-widget-Collapse-header"
  style="padding: 6px; padding-left: 13px; font-size: 14px;"
>
  <span>❌ &nbsp;&nbsp;Failed to load test: '{test_id}'</span>
</div>
"""
