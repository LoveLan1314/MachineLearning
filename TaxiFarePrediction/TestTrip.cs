using System;
using System.Collections.Generic;
using System.Text;

namespace TaxiFarePrediction
{
    static class TestTrip
    {
        internal static readonly TaxiTrip Trip1 = new TaxiTrip
        {
            vendor_id = "VTS",
            rate_code = "1",
            passenger_count = 1,
            trip_distance = 10.33f,
            payment_type = "CSH",
            fare_amount = 0  // predict it. actual = 29.5
        };
    }
}
