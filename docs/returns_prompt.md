Below is an overview of **common return-calculation methods** used by investment funds (focusing on **time-weighted returns** as you mentioned), along with **key formulas** and practical tips to ensure each cash flow is treated fairly based on how long it has actually been invested.

---

## 1. Overview of Common Return Measures

1. **Money-Weighted Return (MWR)**  
   - Also known as the **Internal Rate of Return (IRR)**.  
   - Takes into account the size and timing of cash flows (contributions and withdrawals).  
   - Good for measuring the actual return to the investor because it considers *when* and *how much* money is contributed or withdrawn.  
   - However, it can be heavily influenced by large inflows/outflows.

2. **Time-Weighted Return (TWR)**  
   - Measures the performance of the investment *independent* of the timing and size of external cash flows.  
   - Often used to evaluate the skill of an investment manager or the underlying asset returns *without* penalizing or rewarding for the timing of contributions/withdrawals.  
   - Required by the **Global Investment Performance Standards (GIPS)** for fair performance reporting.

Because you want a **time-weighted return** (“an investment made in December 2022 should carry less weight than one made in January 2022,” and so on), this is the appropriate approach.  

---

## 2. How to Calculate a Time-Weighted Return

The classic method to calculate a TWR is:

1. **Divide the total investment period into sub-periods** based on each external cash flow.
   - For example, if you had contributions/withdrawals in January, March, and September, you would create sub-periods:
     - Sub-period 1: Start of period to end of January (or the day before the cash flow in January)  
     - Sub-period 2: January cash flow to March cash flow  
     - Sub-period 3: March cash flow to September cash flow  
     - Sub-period 4: September cash flow to end of year (or end of the entire measurement period)  
2. **For each sub-period \(i\), calculate the sub-period return \(r_i\)** using the formula (assuming no reinvestment from flows during the sub-period):

   \[
   r_i \;=\; \frac{V_{\text{end},i} - V_{\text{beg},i} - CF_i}{V_{\text{beg},i}}
   \]

   - \(V_{\text{beg},i}\) = market value at the beginning of the sub-period \(i\).  
   - \(V_{\text{end},i}\) = market value at the end of the sub-period \(i\) (before the next cash flow).  
   - \(CF_i\) = net external cash flow *during* sub-period \(i\) (contribution is positive, withdrawal is negative).  

   > *Note:* In practice, some calculations assume the external flow happens at the *start* or *end* of the sub-period. The precise method may vary, but the above is the standard approach for TWR.

3. **Link (geometrically chain) the sub-period returns** over the total period:

   \[
   \text{TWR} = \left[(1 + r_1) \times (1 + r_2) \times \cdots \times (1 + r_n)\right] - 1
   \]

4. **Annualize** (if needed) over \(N\) years:

   \[
   \text{Annualized TWR} = \left[\prod_{i=1}^{n}(1 + r_i)\right]^{\frac{1}{N}} - 1
   \]

   - \(N\) is the number of years in the total measurement period. If it’s 3 years, you would use \(\tfrac{1}{3}\).

### Practical Example

Let’s say:
- You start with \$100,000 on Jan 1, 2022 ( \(V_{beg} = \$100{,}000\) ).
- On June 1, 2022, you add \$20,000 more.
- On December 31, 2022, your account is worth \$130,000.
- On June 1, 2023, you add another \$10,000.
- On December 31, 2023, your account is worth \$150,000.

You would break this into sub-periods:
1. Jan 1, 2022 – June 1, 2022
2. June 1, 2022 – Dec 31, 2022
3. Jan 1, 2023 – June 1, 2023 (assuming you treat Jan 1 value as \$130,000 if no distribution was done on Dec 31)
4. June 1, 2023 – Dec 31, 2023

Each sub-period gets its own return calculation, ignoring the size of the cash flow for that period’s percentage return. Then you chain them for the total TWR.

---

## 3. Ensuring Proportionality and Fair Weight

By using **Time-Weighted Returns**, the “weight” each contribution has is naturally proportional to the **time** it was invested:
- A contribution in January 2022 participates in almost *all* sub-period returns up to the final date.  
- A contribution in December 2022 only participates in the returns from December 2022 onward.

This avoids “giving too much credit” or “penalizing” the manager for large late-year deposits, because TWR measures performance in discrete intervals, net of flows in each sub-period.

---

## 4. Other Considerations

1. **Frequency of Calculation**  
   - You can do TWR on a **daily** or **monthly** basis if you have frequent flows. The more granular, the more accurate, but also more complex.  
   - For many family funds, **monthly** TWR is a good balance between accuracy and practicality.

2. **Reinvestment Assumptions**  
   - TWR inherently assumes that cash flows are not “invested” mid-sub-period, which is why returns are measured from one cash flow to the next, ignoring flows within the period for return calculations.

3. **GIPS Compliance**  
   - If you ever plan to compare or present performance in a standardized way, GIPS guidelines strongly recommend TWR.  
   - GIPS also has specific rules for *partial period returns*, *large external flows*, etc.

4. **Money-Weighted Return (for Investor Experience)**  
   - If at some point you want to see how much *you personally* have earned on the money invested, factoring in the exact *amount and timing* of your investments, you could calculate a money-weighted return (IRR). This is more relevant for your personal “cash-on-cash” return.  
   - But for measuring the performance of the combined family fund in a manager-like manner, TWR is most appropriate.

---

## 5. Putting It All Together

1. **Collect all account values** at each cash flow date (or at each period-end if you are doing monthly TWR).  
2. **Compute sub-period returns** ignoring flows except to split the timeline.  
3. **Chain-link the sub-period returns** to get cumulative TWR for the entire 3-year period.  
4. **Annualize** if desired:
   \[
   \text{Annualized TWR} 
   = \bigl[(1 + r_1) \times \cdots \times (1 + r_n)\bigr]^{\frac{1}{\text{# years}}} - 1.
   \]
5. **Report**:  
   - **Absolute TWR** over 3 years: \(\bigl[(1 + r_1)\cdots(1 + r_n)\bigr] - 1\).  
   - **Annualized TWR**: from the above formula.  
   - **Periodic TWR** (monthly, quarterly, etc.) if you need a more frequent breakdown.

By adhering to the TWR methodology, you ensure each investment (no matter when it was added) is **only measured for the period it was in the account**—this is precisely the fairness you’re seeking.

---

### Key Takeaways

- **Time-Weighted Return (TWR)** is best for manager performance tracking and for ensuring each investment is weighted by the time it actually remains invested.  
- **Money-Weighted Return (IRR)** is best for measuring actual investor experience when there are significant irregular contributions/withdrawals.  
- **Use sub-periods** between each cash flow, calculate each sub-period return, and then **geometrically link**.  
- **Annualize** for a standard year-over-year comparison.  

Following the above formulas and process will allow you to accurately measure the family fund’s returns, ensuring contributions at different times are appropriately accounted for.