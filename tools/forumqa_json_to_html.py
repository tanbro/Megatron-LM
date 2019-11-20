"""将 ForumQA 的 JSON lines 语料文件（可以包括 `inferred` 字段）渲染为 HTML
"""

import argparse
import json
import sys
from itertools import chain

from jinja2 import Template

TEMPLATE_STRING = r'''
<html>
<body>
<ol>
    {{ samples|length }}
    {% for sample in samples %}
    <li>
        <details>
            <summary>
                <h2>{{ sample.title }}</h2>
            </summary>
            {% if sample.tags -%}
            <p>
                [
                {%- for tag in sample.tags -%}
                    {{tag}} {% if not loop.last %},{% endif %}
                {%- endfor -%}
                ]
            </p>
            {%- endif %}
            <article>
                <p>{{ sample.text }}</p>
            </article>
            <!--Answers-->
            <details>
                <summary>Answers({{ sample.answers|length }}):</summary>
                <ol>
                    {% for answer in sample.answers %}
                    <li>
                        <article>
                            <p>{{ answer.text }}</p>
                        </article>
                        </li>
                    {% endfor %}
                </ol>
            </details>
            <!--end Answers-->
            <!--Inferred-->
            <details>
                <summary>Inferred({{ sample.inferred|length }}):</summary>
                <ol>
                    {% for s in sample.inferred %}
                    <li>
                        <article>
                            <p>{{ s }}</p>
                        </article>
                        </li>
                    {% endfor %}
                </ol>
            </details>
            <!--end Inferred-->
        </details>
    </li>
    <hr/>
    {% endfor %}
</ol>
</body>
</html>
'''


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', '-o', type=argparse.FileType('w', encoding='utf8'), default=sys.stdout,
                        help='将渲染后的 HTML 内容输出到这个文件 (default=stdout)')
    parser.add_argument('corpus', nargs='*', type=argparse.FileType('r', encoding='utf8'),
                        help='要进行渲染的 ForumQA 语料文件。如果不指定，该程序将从 stdin 按行读取 JSON 语料')
    args = parser.parse_args()
    return args


def main(args):
    samples = []
    if args.corpus:
        reader = chain.from_iterable(args.corpus)
    else:
        reader = sys.stdin
    for s in reader:
        s = s.strip()
        if s:
            samples.append(json.loads(s))
    template = load_template()
    stream = template.stream(samples=samples)
    for s in stream:
        print(s, file=args.output)


def load_template():
    return Template(TEMPLATE_STRING)


if __name__ == "__main__":
    args = parse_args()
    code = main(args)
    exit(code)
