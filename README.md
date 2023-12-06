# pv-usecase

## Config options

| key                | type                                                 | description                                               | required |
|--------------------|------------------------------------------------------|-----------------------------------------------------------|----------|
| `energy_src_id`    | string                                               | ID of source providing energy data.                       | yes      |
| `weather_src_id`   | string                                               | ID of source providing weather forecast data.             | yes      |
| `logger_level`     | string                                               | `info`, `warning` (default), `error`, `critical`, `debug` | no       |
| `selectors`        | array[object{"name": string, "args": array[string]}] | Define selectors to distinguish between data sources.     | no       |
| `power_td`         | float                                                | Time difference between consecutive power values in min (default:0.17)   | no       |
| `weather_dim`      | integer                                              |                                                           | no       |
| `data_path`        | string                                               | Path to reward and model files. Default: "/opt/data"      | no       |
| `buffer_len`       | integer                                              | Length of replay buffer (default: 48)                     | no       |
| `p_1`              | integer                                              | Power for reward calculation if action==1  (default:1)    | no       |
| `p_0`              | integer                                              | Power for reward calculation if action==0  (default:1)    | no       |
| `history_modus`    | string                                               | `all`, `daylight` (default)                               | no       |
