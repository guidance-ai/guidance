from .._guidance import guidance
from ._ebnf import EBNF
from functools import lru_cache


@lru_cache(maxsize=1)
def iso8601() -> EBNF:
    # https://datatracker.ietf.org/doc/html/rfc3339#appendix-A
    gbnf = """
    !start: iso_date_time

    // Dates

    date_century: DIGIT~2    // 00-99
    date_decade: DIGIT       // 0-9
    date_subdecade: DIGIT    // 0-9
    date_year: date_decade date_subdecade
    date_fullyear: date_century date_year
    date_month: "0" "1".."9" // 01-12
        | "1" "0".."2"
    date_wday: "1".."7"      // 1-7, 1 is Monday, 7 is Sunday
    date_mday: "0" "1".."9"  // TODO: should be 01-28, 01-29, 01-30, 01-31 based on month... NOTE: LEAP YEARS ARE HARD
        | "1" "0".."9"
        | "2" "0".."9"
        | "3" "0".."1"
    date_yday: "00" "1".."9" // 001-365, 001-366 based on year; NOTE: LEAP YEARS ARE HARD
        | "0" "1".."9" DIGIT
        | "1".."2" DIGIT DIGIT
        | "3" "0".."5" DIGIT
        | "36" "0".."6"
    date_week: "0" "1".."9"  // 01-52, 01-53 based on year; NOTE: LEAP YEARS ARE HARD
        | "1".."4" "0".."9"
        | "5" "0".."3"

    datepart_fullyear: [date_century] date_year ["-"]
    datepart_ptyear: "-" [date_subdecade ["-"]]
    datepart_wkyear: datepart_ptyear | datepart_fullyear

    dateopt_century: "-" | date_century
    dateopt_fullyear: "-" | datepart_fullyear
    dateopt_year: "-" | (date_year ["-"])
    dateopt_month: "-" | (date_month ["-"])
    dateopt_week: "-" | (date_week ["-"])

    datespec_full: datepart_fullyear date_month ["-"] date_mday
    datespec_year: date_century | dateopt_century date_year
    datespec_month: "-" dateopt_year date_month ["-"]? date_mday?
    datespec_mday: "--" dateopt_month date_mday
    datespec_week: datepart_wkyear "W" (date_week | dateopt_week date_wday)
    datespec_wday: "---" date_wday
    datespec_yday: dateopt_fullyear date_yday

    date: datespec_full | datespec_year | datespec_month | datespec_mday | datespec_week | datespec_wday | datespec_yday

    // Times

    time_hour: "0".."1" DIGIT       // 00-24
        | "2" "0".."4"
    time_minute: "0".."5" DIGIT     // 00-59
    time_second: "0".."5" DIGIT     // 00-58, 00-59, 00-60 based on leap-second rules; NOTE: LEAP YEARS ARE HARD
        | "60"
    time_fraction: ["," | "."] DIGIT+

    time_numoffset: ("+" | "-") time_hour [":" time_minute]
    time_zone: "Z" | time_numoffset

    timeopt_hour: "-" | (time_hour [":"])
    timeopt_minute: "-" | (time_minute [":"])

    timespec_hour: time_hour [":" time_minute [":" time_second]]
    timespec_minute: timeopt_hour time_minute [":" time_second]
    timespec_second: "-" timeopt_minute time_second
    timespec_base: timespec_hour | timespec_minute | timespec_second

    time: timespec_base [time_fraction] [time_zone]

    iso_date_time: date "T" time

    // Durations

    dur_second: DIGIT+ "S"
    dur_minute: DIGIT+ "M" [dur_second]
    dur_hour: DIGIT+ "H" [dur_minute]
    dur_time: "T" (dur_hour | dur_minute | dur_second)
    dur_day: DIGIT+ "D"
    dur_week: DIGIT+ "W"
    dur_month: DIGIT+ "M" [dur_day]
    dur_year: DIGIT+ "Y" [dur_month]
    dur_date: (dur_day | dur_month | dur_year) [dur_time]

    duration: "P" (dur_date | dur_time | dur_week)

    // Periods

    period_explicit: iso_date_time "/" iso_date_time
    period_start: iso_date_time "/" duration
    period_end: duration "/" iso_date_time

    period: period_explicit | period_start | period_end

    // Terminals
    
    DIGIT: /[0-9]/
    """
    return EBNF.from_grammar_string(gbnf)


@guidance(stateless=True, cache=True)
def date_time(lm, name=None):
    return lm + iso8601().build(name=name, start="iso_date_time")

@guidance(stateless=True, cache=True)
def time(lm, name=None):
    return lm + iso8601().build(name=name, start="time")

@guidance(stateless=True, cache=True)
def date(lm, name=None):
    return lm + iso8601().build(name=name, start="date")

@guidance(stateless=True, cache=True)
def duration(lm, name=None):
    return lm + iso8601().build(name=name, start="duration")