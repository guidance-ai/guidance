import pytest

import pydantic
import datetime

from guidance.library._iso8601 import date_time
from ...utils import generate_and_check, check_match_failure


class TestURI:
    @pytest.mark.parametrize(
        "target_obj",
        [
            '2006-10-09T04:46:51',
            '2017-03-30T05:02:37',
            '2014-05-02T22:28:14+05:30',
            '2014-05-23T04:11:58-08:00',
            '2008-11-16T13:38:40+05:30',
            '2003-08-15T07:32:10+05:30',
            '2016-12-08T05:31:49',
            '2006-03-26T00:10:08-08:00',
            '2003-12-12T13:16:44-05:00',
            '2002-12-23T04:05:30+09:00',
            '2002-09-27T18:15:09+00:00',
            '2004-12-30T15:56:08+09:00',
            '2010-01-04T16:48:55-08:00',
            '2000-11-05T19:49:29+00:00',
            '2015-10-12T09:32:01',
            '2007-02-18T01:03:42',
            '2009-04-22T01:01:37-08:00',
            '2001-07-06T22:29:11',
            '2001-04-16T05:28:47+00:00'
        ],
    )
    def test_datetime(self, target_obj):
        # First sanity check the target object
        pydantic.TypeAdapter(datetime.datetime).validate_python(target_obj)
        
        generate_and_check(
            grammar_callable=date_time,
            test_string=target_obj,
        )

    @pytest.mark.parametrize(
        "bad_obj",
        [
            '2003-11-20T00',                    # Missing minute and second components
            '2005-10-29T00:00:00Z07',           # Malformed timezone (extra characters)
            '2005-00-00T12:30:30',              # Invalid month and day (zero)
            '2001-02-29T12:00:00',              # Non-existent leap day for the year 2001
            '2011-06-03T00',                    # Missing minute and second components
            '-2019-03-14T00:00:00',             # Negative year
            '2008-13-32T12:30:30',              # Invalid month and day
            '2015-10-31T25:61:62',              # Invalid hour, minute, and second
            '2001/02/19 T 00-00-00',            # Malformed with incorrect separators
            '2016-13-32T12:30:30',              # Invalid month and day
            '2002-04-16T00:00:00Z07',           # Malformed timezone (extra characters)
            '2005-00-00T12:30:30',              # Invalid month and day (zero)
            '2003-07-23T00:00:00+25:00',        # Invalid timezone offset
            '2013-07-05T00:00:00Z07',           # Malformed timezone (extra characters)
            '2005-00-00T12:30:30',              # Invalid month and day (zero)
            '2016-12-04T00:00:00Z07',           # Malformed timezone (extra characters)
            '2013-02-14T25:61:62',              # Invalid hour, minute, and second
            '2002/04/12 T 00-00-00',            # Malformed with incorrect separators
            '2007-01-01T00',                    # Missing minute and second components
            '2005-00-00T12:30:30'               # Invalid month and day (zero)
        ],
    )
    def test_bad_datetime(self, bad_obj):
        # First sanity check the target object
        with pytest.raises(pydantic.ValidationError):
            pydantic.TypeAdapter(datetime.datetime).validate_python(bad_obj)

        check_match_failure(
            bad_string=bad_obj,
            grammar=date_time(),
        )